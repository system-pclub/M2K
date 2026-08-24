//===-- Executor.h ----------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class to perform actual execution, hides implementation details from external
// interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_EXECUTOR_H
#define KLEE_EXECUTOR_H

#include "ExecutionState.h"
#include "UserSearcher.h"
#include "LoopInfo.h"

#include "klee/ADT/RNG.h"
#include "klee/Core/BranchTypes.h"
#include "klee/Core/Interpreter.h"
#include "klee/Core/TerminationTypes.h"
#include "klee/Expr/ArrayCache.h"
#include "klee/Expr/ArrayExprOptimizer.h"
#include "klee/Module/Cell.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"
#include "klee/System/Time.h"

#include "llvm/IR/Module.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct KTest;

namespace llvm {
  class BasicBlock;
  class BranchInst;
  class CallInst;
  class LandingPadInst;
  class Constant;
  class ConstantExpr;
  class Function;
  class GlobalValue;
  class Instruction;
  class LLVMContext;
  class DataLayout;
  class Twine;
  class Value;
  class Loop;
}

namespace klee {
class Array;
struct Cell;
class ExecutionState;
class ExternalDispatcher;
class Expr;
class InstructionInfoTable;
class KCallable;
struct KFunction;
struct KInstruction;
class KInstIterator;
class KModule;
class MemoryManager;
class MemoryObject;
class ObjectState;
class ExecutionTree;
class Searcher;
class SeedInfo;
class SpecialFunctionHandler;
struct StackFrame;
class StatsTracker;
class TimingSolver;
class TreeStreamWriter;
class MergeHandler;
class MergingSearcher;
template <class T> class ref;

/// \todo Add a context object to keep track of data only live
/// during an instruction step. Should contain addedStates,
/// removedStates, and haltExecution, among others.

class Executor : public Interpreter {
  friend class OwningSearcher;
  friend class WeightedRandomSearcher;
  friend class SpecialFunctionHandler;
  friend class StatsTracker;
  friend class MergeHandler;
  friend class ObjectState;
  friend klee::Searcher *klee::constructUserSearcher(Executor &executor);

public:
  typedef std::pair<ExecutionState*,ExecutionState*> StatePair;

  /// The random number generator.
  RNG theRNG;

private:
  std::unique_ptr<KModule> kmodule;
  InterpreterHandler *interpreterHandler;
  Searcher *searcher;

  ExternalDispatcher *externalDispatcher;
  std::unique_ptr<TimingSolver> solver;
  std::unique_ptr<MemoryManager> memory;
  std::set<ExecutionState*, ExecutionStateIDCompare> states;
  StatsTracker *statsTracker;
  TreeStreamWriter *pathWriter, *symPathWriter;
  SpecialFunctionHandler *specialFunctionHandler;
  TimerGroup timers;
  std::unique_ptr<ExecutionTree> executionTree;

  /// Used to track states that have been added during the current
  /// instructions step. 
  /// \invariant \ref addedStates is a subset of \ref states. 
  /// \invariant \ref addedStates and \ref removedStates are disjoint.
  std::vector<ExecutionState *> addedStates;
  /// Used to track states that have been removed during the current
  /// instructions step. 
  /// \invariant \ref removedStates is a subset of \ref states. 
  /// \invariant \ref addedStates and \ref removedStates are disjoint.
  std::vector<ExecutionState *> removedStates;

  /// When non-empty the Executor is running in "seed" mode. The
  /// states in this map will be executed in an arbitrary order
  /// (outside the normal search interface) until they terminate. When
  /// the states reach a symbolic branch then either direction that
  /// satisfies one or more seeds will be added to this map. What
  /// happens with other states (that don't satisfy the seeds) depends
  /// on as-yet-to-be-determined flags.
  std::map<ExecutionState*, std::vector<SeedInfo> > seedMap;

  /// Map of globals to their representative memory object.
  std::map<const llvm::GlobalValue*, MemoryObject*> globalObjects;

  /// Map of globals to their bound address. This also includes
  /// globals that have no representative object (e.g. aliases).
  std::map<const llvm::GlobalValue*, ref<ConstantExpr>> globalAddresses;

  /// Map of legal function addresses to the corresponding Function.
  /// Used to validate and dereference function pointers.
  std::unordered_map<std::uint64_t, llvm::Function*> legalFunctions;

  /// When non-null the bindings that will be used for calls to
  /// klee_make_symbolic in order replay.
  const struct KTest *replayKTest;

  /// When non-null a list of branch decisions to be used for replay.
  const std::vector<bool> *replayPath;

  /// The index into the current \ref replayKTest or \ref replayPath
  /// object.
  unsigned replayPosition;

  /// When non-null a list of "seed" inputs which will be used to
  /// drive execution.
  const std::vector<struct KTest *> *usingSeeds;  

  /// Disables forking, instead a random path is chosen. Enabled as
  /// needed to control memory usage. \see fork()
  bool atMemoryLimit;

  /// Disables forking, set by client. \see setInhibitForking()
  bool inhibitForking;

  /// Signals the executor to halt execution at the next instruction
  /// step.
  bool haltExecution;  

  /// Whether implied-value concretization is enabled. Currently
  /// false, it is buggy (it needs to validate its writes).
  bool ivcEnabled;

  /// The maximum time to allow for a single core solver query.
  /// (e.g. for a single STP query)
  time::Span coreSolverTimeout;

  /// Maximum time to allow for a single instruction.
  time::Span maxInstructionTime;

  /// Assumes ownership of the created array objects
  ArrayCache arrayCache;

  /// File to print executed instructions to
  std::unique_ptr<llvm::raw_ostream> debugInstFile;

  // @brief Buffer used by logBuffer
  std::string debugBufferString;

  // @brief buffer to store logs before flushing to file
  llvm::raw_string_ostream debugLogBuffer;

  /// Optimizes expressions
  ExprOptimizer optimizer;

  /// Points to the merging searcher of the searcher chain,
  /// `nullptr` if merging is disabled
  MergingSearcher *mergingSearcher = nullptr;

  /// Typeids used during exception handling
  std::vector<ref<Expr>> eh_typeids;

  llvm::Module *mainModule;
  std::vector<std::string> kernelNames;
  std::string outputDir;


  std::map<ref<Expr>, std::pair<std::string, bool>> sharedAddresses; // address->(name, isDynamic)

  std::vector<uint64_t> deletedAddresses;
  std::map<std::string, bool> isIntArgSigned;
  static int kernelConfigCounter;

  std::map<unsigned, Interpreter::ParameterValue> argumentValues;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> parCons; // equal parameters
  std::map<std::string, std::pair<int, int>> symRangeCons; // value ranges of expr
  std::map<std::string, unsigned> promptCons; // num_tokens limit
  bool isLoopCheck = false;
  std::map<llvm::BranchInst*, llvm::Instruction*> trampolineBrs;

  void saveLoopInfo(const llvm::Loop *L, ExecutionState &state);


  /// Return the typeid corresponding to a certain `type_info`
  ref<ConstantExpr> getEhTypeidFor(ref<Expr> type_info);

  llvm::Function* getTargetFunction(llvm::Value *calledVal);

  std::pair<int, const MemoryObject *> findInductionVariable(ExecutionState &state, llvm::Instruction *ii, KLoopInfo * loopInfo);

  void setElementSize(ExecutionState &state, SymArrayMemoryObject *symmo);

  MemoryObject* loadTensorConfig(ExecutionState *state, ParameterValue pval, std::string symName, unsigned argSize);
  
  void executeInstruction(ExecutionState &state, KInstruction *ki);

  void run(ExecutionState &initialState);

  // Given a concrete object in our [klee's] address space, add it to 
  // objects checked code can reference.
  MemoryObject *addExternalObject(ExecutionState &state, void *addr, 
                                  unsigned size, bool isReadOnly);

  void initializeGlobalAlias(const llvm::Constant *c);
  void initializeGlobalObject(ExecutionState &state, ObjectState *os, 
			      const llvm::Constant *c,
			      unsigned offset);
  void initializeGlobals(ExecutionState &state);
  void allocateGlobalObjects(ExecutionState &state);
  void initializeGlobalAliases();
  void initializeGlobalObjects(ExecutionState &state);

  void stepInstruction(ExecutionState &state);
  void updateStates(ExecutionState *current);
  void transferToBasicBlock(llvm::BasicBlock *dst, 
			    llvm::BasicBlock *src,
			    ExecutionState &state);

  void callExternalFunction(ExecutionState &state,
                            KInstruction *target,
                            KCallable *callable,
                            std::vector< ref<Expr> > &arguments);

  ObjectState *bindObjectInState(ExecutionState &state, const MemoryObject *mo,
                                 bool isLocal, const Array *array = 0);

  /// Resolve a pointer to the memory objects it could point to the
  /// start of, forking execution when necessary and generating errors
  /// for pointers to invalid locations (either out of bounds or
  /// address inside the middle of objects).
  ///
  /// \param results[out] A list of ((MemoryObject,ObjectState),
  /// state) pairs for each object the given address can point to the
  /// beginning of.
  typedef std::vector< std::pair<std::pair<const MemoryObject*, const ObjectState*>, 
                                 ExecutionState*> > ExactResolutionList;
  void resolveExact(ExecutionState &state,
                    ref<Expr> p,
                    ExactResolutionList &results,
                    const std::string &name);
  
  unsigned getTypeBits(llvm::Type *returnType);

  unsigned getTensorTypeBytes(ref<Expr> scalarType);

  ref<Expr> getScalarTypeByStr(std::string typeStr);

  ref<Expr> getAccumalateTensorType(ref<Expr> scalarType);

  ref<Expr> convertFP2Int(ExecutionState &state, ref<Expr> arg, bool isSigned, Expr::Width resultType);

  void computeTensorIteratorShape(ExecutionState &state);

  void computeTensorIteratorStrides(ExecutionState &state, SymArrayMemoryObject *symmo);

  std::string getSymName(ExecutionState &state, const MemoryObject *mo, ref<Expr> address);

  SymArrayMemoryObject* createSymArray(ExecutionState &state, ref<Expr> address, const MemoryObject *destMO, const std::string& funcName, const std::string& tensorNamePrefix);


  bool isReadSymArrayWithSymIndex(ExecutionState &state, ref<Expr> expr);

  std::pair<const Array *, ref<Expr>> createSizeSymbol(ExecutionState &state, const std::string &sizeName, bool isTotal=false);
  
  ref<Expr> symbolizeIndex(ExecutionState &state, const std::string &name, const MemoryObject *mo, bool isSigned=true, ref<Expr> initialVal=nullptr);
  
  ref<Expr> createIndexRangeExpr(ExecutionState &state, const std::string &name, uint64_t size, bool isLocal, ref<Expr> rangeMin, ref<Expr> rangeMax, bool compareSigned=false, const MemoryObject *mo=nullptr);

  void validateKernelConfig(ExecutionState &state, ref<Expr> config, ref<Expr> maxValueExpr=nullptr);
  void initializeCudaKernelConfig(ExecutionState &state, ref<Expr> gridxExpr, ref<Expr> gridyExpr, ref<Expr> gridzExpr, 
                                          ref<Expr> blockxExpr, ref<Expr> blockyExpr, ref<Expr> blockzExpr, ref<Expr> sharedMemSizeExpr);

  void addBoundConstraintForArg(ExecutionState &state, KInstruction *ki);

  double parseCoefficient(const std::string& token);

  ref<Expr> getParByIndex(std::tuple<int, int, int> item, ExecutionState *state, std::vector<ref<Expr>>& arguments);

  bool isNumber(const std::string& s);

  void modifyArrayBound(ExecutionState &state, ref<Expr> value, SymArrayMemoryObject *symmo, KInstruction *i);

  /// Allocate and bind a new object in a particular state. NOTE: This
  /// function may fork.
  ///
  /// \param isLocal Flag to indicate if the object should be
  /// automatically deallocated on function return (this also makes it
  /// illegal to free directly).
  ///
  /// \param target Value at which to bind the base address of the new
  /// object.
  ///
  /// \param reallocFrom If non-zero and the allocation succeeds,
  /// initialize the new object from the given one and unbind it when
  /// done (realloc semantics). The initialized bytes will be the
  /// minimum of the size of the old and new objects, with remaining
  /// bytes initialized as specified by zeroMemory.
  ///
  /// \param allocationAlignment If non-zero, the given alignment is
  /// used. Otherwise, the alignment is deduced via
  /// Executor::getAllocationAlignment
  void executeAlloc(ExecutionState &state,
                    ref<Expr> size,
                    bool isLocal,
                    KInstruction *target,
                    bool zeroMemory=false,
                    const ObjectState *reallocFrom=0,
                    size_t allocationAlignment=0,
                    ref<Expr> cudaAddress = nullptr);

  /// Free the given address with checking for errors. If target is
  /// given it will be bound to 0 in the resulting states (this is a
  /// convenience for realloc). Note that this function can cause the
  /// state to fork and that \ref state cannot be safely accessed
  /// afterwards.
  void executeFree(ExecutionState &state,
                   ref<Expr> address,
                   KInstruction *target = 0);

  /// Serialize a landingpad instruction so it can be handled by the
  /// libcxxabi-runtime
  MemoryObject *serializeLandingpad(ExecutionState &state,
                                    const llvm::LandingPadInst &lpi,
                                    bool &stateTerminated);

  /// Unwind the given state until it hits a landingpad. This is used
  /// for exception handling.
  void unwindToNextLandingpad(ExecutionState &state);

  void executeCall(ExecutionState &state, 
                   KInstruction *ki,
                   llvm::Function *f,
                   std::vector< ref<Expr> > &arguments);

  void executeCudaMemcpy(ExecutionState &state,
                                      ref<Expr> desAddress,
                                      ref<Expr> srcAddress,
                                      ref<Expr> size,
                                      KInstruction *target);

  bool exprContainsConstant(ref<Expr> expr, uint64_t target);
  // do address resolution / object binding / out of bounds checking
  // and perform the operation
  void executeMemoryOperation(ExecutionState &state,
                              bool isWrite,
                              ref<Expr> address,
                              ref<Expr> value /* undef if read */,
                              KInstruction *target /* undef if write */, unsigned loadBytes=0, ref<Expr> selectAddress=0);

  void storeAddressInLoop(ExecutionState &state, KInstruction *ki, ref<Expr> address, const MemoryObject *mo, bool isWrite);
  void checkDataRace(ExecutionState &state, KLoopInfo *loopInfo);

  void executeMakeSymbolic(ExecutionState &state, const MemoryObject *mo,
                           const std::string &name);

  /// Create a new state where each input condition has been added as
  /// a constraint and return the results. The input state is included
  /// as one of the results. Note that the output vector may include
  /// NULL pointers for states which were unable to be created.
  void branch(ExecutionState &state, const std::vector<ref<Expr>> &conditions,
              std::vector<ExecutionState *> &result, BranchType reason);

  /// Fork current and return states in which condition holds / does
  /// not hold, respectively. One of the states is necessarily the
  /// current state, and one of the states may be null.
  StatePair fork(ExecutionState &current, ref<Expr> condition, bool isInternal,
                 BranchType reason);

  // If the MaxStatic*Pct limits have been reached, concretize the condition and
  // return it. Otherwise, return the unmodified condition.
  ref<Expr> maxStaticPctChecks(ExecutionState &current, ref<Expr> condition);

  /// Add the given (boolean) condition as a constraint on state. This
  /// function is a wrapper around the state's addConstraint function
  /// which also manages propagation of implied values,
  /// validity checks, and seed patching.
  void addConstraint(ExecutionState &state, ref<Expr> condition);

  // Called on [for now] concrete reads, replaces constant with a symbolic
  // Used for testing.
  ref<Expr> replaceReadWithSymbolic(ExecutionState &state, ref<Expr> e);

  const Cell& eval(KInstruction *ki, unsigned index, 
                   ExecutionState &state) const;

  Cell& getArgumentCell(ExecutionState &state,
                        KFunction *kf,
                        unsigned index) {
    return state.stack.back().locals[kf->getArgRegister(index)];
  }

  Cell& getDestCell(ExecutionState &state,
                    KInstruction *target) {
    return state.stack.back().locals[target->dest];
  }

  void bindLocal(KInstruction *target, 
                 ExecutionState &state, 
                 ref<Expr> value);
  void bindArgument(KFunction *kf, 
                    unsigned index,
                    ExecutionState &state,
                    ref<Expr> value);

  /// Evaluates an LLVM constant expression.  The optional argument ki
  /// is the instruction where this constant was encountered, or NULL
  /// if not applicable/unavailable.
  ref<klee::ConstantExpr> evalConstantExpr(const llvm::ConstantExpr *c,
					   const KInstruction *ki = NULL);

  /// Evaluates an LLVM constant.  The optional argument ki is the
  /// instruction where this constant was encountered, or NULL if
  /// not applicable/unavailable.
  ref<klee::ConstantExpr> evalConstant(const llvm::Constant *c,
				       const KInstruction *ki = NULL);

  /// Return a unique constant value for the given expression in the
  /// given state, if it has one (i.e. it provably only has a single
  /// value). Otherwise return the original expression.
  ref<Expr> toUnique(const ExecutionState &state, ref<Expr> &e);

  /// Return a constant value for the given expression.  If \param concretize is
  /// true, the expression is forced to be a constant by adding a constraint if
  /// necessary. Note that this function breaks completeness and
  /// should generally be avoided.
  ///
  /// \param reason A documentation string stating the reason for concretization
  ref<klee::ConstantExpr> toConstant(ExecutionState &state, ref<Expr> e,
                                     const std::string &reason,
                                     bool concretize = true);

  /// Evaluate the given expression under each seed, and return the
  /// first one that results in a constant, if such a seed exist.  Otherwise,
  /// return the non-constant evaluation of the expression under one of the
  /// seeds.
  ref<klee::ConstantExpr> getValueFromSeeds(ExecutionState &state, ref<Expr> e);

  /// Bind a constant value for e to the given target. NOTE: This
  /// function may fork state if the state has multiple seeds.
  void executeGetValue(ExecutionState &state, ref<Expr> e, KInstruction *target);

  /// Get textual information regarding a memory address.
  std::string getAddressInfo(ExecutionState &state, ref<Expr> address) const;

  // Determines the \param lastInstruction of the \param state which is not KLEE
  // internal and returns its InstructionInfo
  const InstructionInfo & getLastNonKleeInternalInstruction(const ExecutionState &state,
      llvm::Instruction** lastInstruction);

  void storeBuggyQuery(ExecutionState &state, StateTerminationType reason, std::string message, int sourceLine, unsigned assemblyLine);
  /// Remove state from queue and delete state. This function should only be
  /// used in the termination functions below.
  void terminateState(ExecutionState &state, StateTerminationType reason);

  /// Call exit handler and terminate state normally
  /// (end of execution path)
  void terminateStateOnExit(ExecutionState &state);

  /// Call exit handler and terminate state early
  /// (e.g. due to state merging or memory pressure)
  void terminateStateEarly(ExecutionState &state, const llvm::Twine &message,
                           StateTerminationType reason);

  /// Call exit handler and terminate state early
  /// (e.g. caused by the applied algorithm as in state merging or replaying)
  void terminateStateEarlyAlgorithm(ExecutionState &state,
                                    const llvm::Twine &message,
                                    StateTerminationType reason);

  /// Call exit handler and terminate state early
  /// (e.g. due to klee_silent_exit issued by user)
  void terminateStateEarlyUser(ExecutionState &state,
                               const llvm::Twine &message);

  /// Call error handler and terminate state in case of errors.
  /// The underlying function of all error-handling termination functions
  /// below. This function should only be used in the termination functions
  /// below.
  void terminateStateOnError(ExecutionState &state,
                                    const llvm::Twine &message,
                                    StateTerminationType reason,
                                    const llvm::Twine &longMessage = "",
                                    const char *suffix = nullptr);

  /// Call error handler and terminate state in case of program errors
  /// (e.g. free()ing globals, out-of-bound accesses)
  void terminateStateOnProgramError(ExecutionState &state,
                                    const llvm::Twine &message,
                                    StateTerminationType reason,
                                    const llvm::Twine &longMessage = "",
                                    const char *suffix = nullptr);

  /// Call error handler and terminate state in case of execution errors
  /// (things that should not be possible, like illegal instruction or
  /// unlowered intrinsic, or unsupported intrinsics, like inline assembly)
  void terminateStateOnExecError(
      ExecutionState &state, const llvm::Twine &message,
      StateTerminationType = StateTerminationType::Execution);

  /// Call error handler and terminate state in case of solver errors
  /// (solver error or timeout)
  void terminateStateOnSolverError(ExecutionState &state,
                                   const llvm::Twine &message);

  /// Call error handler and terminate state for user errors
  /// (e.g. wrong usage of klee.h API)
  void terminateStateOnUserError(ExecutionState &state,
                                 const llvm::Twine &message,
                                 bool writeErr = true);

  /// bindModuleConstants - Initialize the module constant table.
  void bindModuleConstants();

  template <typename TypeIt>
  void computeOffsetsSeqTy(KGEPInstruction *kgepi,
                           ref<ConstantExpr> &constantOffset, uint64_t index,
                           const TypeIt it);

  template <typename TypeIt>
  void computeOffsets(KGEPInstruction *kgepi, TypeIt ib, TypeIt ie);

  /// bindInstructionConstants - Initialize any necessary per instruction
  /// constant values.
  void bindInstructionConstants(KInstruction *KI);

  void doImpliedValueConcretization(ExecutionState &state,
                                    ref<Expr> e,
                                    ref<ConstantExpr> value);

  /// check memory usage and terminate states when over threshold of -max-memory + 100MB
  /// \return true if below threshold, false otherwise (states were terminated)
  bool checkMemoryUsage();

  /// check if branching/forking is allowed
  bool branchingPermitted(const ExecutionState &state) const;

  void printDebugInstructions(ExecutionState &state);
  void doDumpStates();

  /// Only for debug purposes; enable via debugger or klee-control
  void dumpStates();
  void dumpExecutionTree();

public:
  Executor(llvm::LLVMContext &ctx, const InterpreterOptions &opts,
      InterpreterHandler *ie);
  virtual ~Executor();

  const InterpreterHandler& getHandler() {
    return *interpreterHandler;
  }

  void setPathWriter(TreeStreamWriter *tsw) override { pathWriter = tsw; }

  void setSymbolicPathWriter(TreeStreamWriter *tsw) override {
    symPathWriter = tsw;
  }

  void setReplayKTest(const struct KTest *out) override {
    assert(!replayPath && "cannot replay both buffer and path");
    replayKTest = out;
    replayPosition = 0;
  }

  void setReplayPath(const std::vector<bool> *path) override {
    assert(!replayKTest && "cannot replay both buffer and path");
    replayPath = path;
    replayPosition = 0;
  }

  llvm::Module *setModule(std::vector<std::unique_ptr<llvm::Module>> &modules,
                          const ModuleOptions &opts) override;
  
  void setMainModule(llvm::Module *module) override {
    mainModule = module;
  }

  void setKernelNames(std::vector<std::string> &_kernelNames) override {
    kernelNames = _kernelNames;
  }

  void setOutputDir(const std::string &_dir) override {
    outputDir = _dir;
  }

  void setArgumentValues(std::map<unsigned, Interpreter::ParameterValue> &_argumentValues) {
    argumentValues = _argumentValues;
  }

  void setArgumentCons(std::map<std::string, std::vector<std::tuple<int, int, int>>> equals) {
    parCons = equals;
  }

  void setVRangeCons(std::map<std::string, std::pair<int, int>> ranges) {
    symRangeCons = ranges;
  }

  void setPromptCons(std::map<std::string, unsigned> _promptCons) {
    promptCons = _promptCons;
  }

  llvm::Function* getFunctionByName(const std::string &fname) {
    return kmodule->module.get()->getFunction(fname);
  }

  void useSeeds(const std::vector<struct KTest *> *seeds) override {
    usingSeeds = seeds;
  }

  void initializeLoopInfo(llvm::Module *M, ExecutionState &state);

  void findTrampolineBranches(llvm::Module *M);

  bool isMemFuncName(const std::string &funcName);

  void runKernelFunction(llvm::Function *f, bool hasLaunchKernel) override;

  void runFunctionAsMain(llvm::Function *f, int argc, char **argv,
                         char **envp) override;

  /*** Runtime options ***/

  void setHaltExecution(bool value) override { haltExecution = value; }

  void setInhibitForking(bool value) override { inhibitForking = value; }

  void prepareForEarlyExit() override;

  /*** State accessor methods ***/

  unsigned getPathStreamID(const ExecutionState &state) override;

  unsigned getSymbolicPathStreamID(const ExecutionState &state) override;

  void getConstraintLog(const ExecutionState &state, std::string &res,
                        Interpreter::LogType logFormat =
                            Interpreter::STP) override;

  bool getSymbolicSolution(
      const ExecutionState &state,
      std::vector<std::pair<std::string, std::vector<unsigned char>>> &res)
      override;

  void getCoveredLines(const ExecutionState &state,
                       std::map<const std::string *, std::set<unsigned>> &res)
      override;

  Expr::Width getWidthForLLVMType(llvm::Type *type) const;
  size_t getAllocationAlignment(const llvm::Value *allocSite) const;

  /// Returns the errno location in memory of the state
  int *getErrnoLocation(const ExecutionState &state) const;

  MergingSearcher *getMergingSearcher() const { return mergingSearcher; };
  void setMergingSearcher(MergingSearcher *ms) { mergingSearcher = ms; };
};

} // namespace klee

#endif /* KLEE_EXECUTOR_H */
