//===-- Executor.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Executor.h"

#include "AddressSpace.h"
#include "Context.h"
#include "CoreStats.h"
#include "ExecutionState.h"
#include "ExecutionTree.h"
#include "ExternalDispatcher.h"
#if LLVM_VERSION_CODE <= LLVM_VERSION(14, 0)
#include "GetElementPtrTypeIterator.h"
#endif
#include "ImpliedValue.h"
#include "Memory.h"
#include "MemoryManager.h"
#include "Searcher.h"
#include "SeedInfo.h"
#include "SpecialFunctionHandler.h"
#include "StatsTracker.h"
#include "TimingSolver.h"
#include "UserSearcher.h"

#include "klee/ADT/KTest.h"
#include "klee/ADT/RNG.h"
#include "klee/Config/Version.h"
#include "klee/Core/Interpreter.h"
#include "klee/Expr/ArrayExprOptimizer.h"
#include "klee/Expr/Assignment.h"
#include "klee/Expr/Expr.h"
#include "klee/Expr/ExprPPrinter.h"
#include "klee/Expr/ExprSMTLIBPrinter.h"
#include "klee/Expr/ExprUtil.h"
#include "klee/KDAlloc/kdalloc.h"
#include "klee/Module/Cell.h"
#include "klee/Module/InstructionInfoTable.h"
#include "klee/Module/KCallable.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"
#include "klee/Solver/Common.h"
#include "klee/Solver/SolverCmdLine.h"
#include "klee/Solver/SolverStats.h"
#include "klee/Statistics/TimerStatIncrementer.h"
#include "klee/Support/Casting.h"
#include "klee/Support/ErrorHandling.h"
#include "klee/Support/FileHandling.h"
#include "klee/Support/ModuleUtil.h"
#include "klee/Support/OptionCategories.h"
#include "klee/System/MemoryUsage.h"
#include "klee/System/Time.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#if LLVM_VERSION_CODE >= LLVM_VERSION(15, 0)
#include "llvm/IR/GetElementPtrTypeIterator.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <cxxabi.h>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <vector>
#include <regex>
#include <filesystem>

using namespace llvm;
using namespace klee;

namespace klee {
cl::OptionCategory DebugCat("Debugging options",
                            "These are debugging options.");

cl::OptionCategory ExtCallsCat("External call policy options",
                               "These options impact external calls.");

cl::OptionCategory SeedingCat(
    "Seeding options",
    "These options are related to the use of seeds to start exploration.");

cl::OptionCategory
    TerminationCat("State and overall termination options",
                   "These options control termination of the overall KLEE "
                   "execution and of individual states.");

cl::OptionCategory TestGenCat("Test generation options",
                              "These options impact test generation.");

cl::opt<std::string> MaxTime(
    "max-time",
    cl::desc("Halt execution after the specified duration.  "
             "Set to 0s to disable (default=0s)"),
    cl::init("0s"),
    cl::cat(TerminationCat));

cl::opt<bool>
  AllowSymLoop("sym-loop",
                 cl::desc("symbolize loop index (default=true)."),
		 cl::init(true));

/*** Misc options ***/
cl::opt<bool> SingleObjectResolution(
    "single-object-resolution",
    cl::desc("Try to resolve memory reads/writes to single objects "
             "when offsets are symbolic (default=false)"),
    cl::init(false), cl::cat(MiscCat));
} // namespace klee

namespace {

/*** Test generation options ***/

cl::opt<bool> DumpStatesOnHalt(
    "dump-states-on-halt",
    cl::init(true),
    cl::desc("Dump test cases for all active states on exit (default=true)"),
    cl::cat(TestGenCat));

cl::opt<bool> OnlyOutputStatesCoveringNew(
    "only-output-states-covering-new",
    cl::init(false),
    cl::desc("Only output test cases covering new code (default=false)"),
    cl::cat(TestGenCat));

cl::opt<bool> EmitAllErrors(
    "emit-all-errors", cl::init(false),
    cl::desc("Generate tests cases for all errors "
             "(default=false, i.e. one per (error,instruction) pair)"),
    cl::cat(TestGenCat));


/* Constraint solving options */

cl::opt<unsigned> MaxSymArraySize(
    "max-sym-array-size",
    cl::desc(
        "If a symbolic array exceeds this size (in bytes), symbolic addresses "
        "into this array are concretized.  Set to 0 to disable (default=0)"),
    cl::init(0),
    cl::cat(SolvingCat));

cl::opt<bool>
    SimplifySymIndices("simplify-sym-indices",
                       cl::init(false),
                       cl::desc("Simplify symbolic accesses using equalities "
                                "from other constraints (default=false)"),
                       cl::cat(SolvingCat));

cl::opt<bool>
    EqualitySubstitution("equality-substitution", cl::init(true),
                         cl::desc("Simplify equality expressions before "
                                  "querying the solver (default=true)"),
                         cl::cat(SolvingCat));


/*** External call policy options ***/

enum class ExternalCallPolicy {
  None,       // No external calls allowed
  Concrete,   // Only external calls with concrete arguments allowed
  All,        // All external calls allowed, symbolic arguments concretized
  OverApprox, // All external calls allowed, symbolic inputs are not constrained by the call
};

cl::opt<ExternalCallPolicy> ExternalCalls(
    "external-calls",
    cl::desc("Specify the external call policy"),
    cl::values(
        clEnumValN(
            ExternalCallPolicy::None, "none",
            "No external function calls are allowed.  Note that KLEE always "
            "allows some external calls with concrete arguments to go through "
            "(in particular printf and puts), regardless of this option."),
        clEnumValN(ExternalCallPolicy::Concrete, "concrete",
                   "Only external function calls with concrete arguments are "
                   "allowed (default)"),
        clEnumValN(ExternalCallPolicy::All, "all",
                   "All external function calls are allowed.  This concretizes "
                   "any symbolic arguments in calls to external functions."),
        clEnumValN(ExternalCallPolicy::OverApprox, "over-approx",
                   "All external function calls are allowed.  This concretizes "
                   "any symbolic arguments in calls to external functions but"
                   "the concretization is ignored after the call (see docs).")),
    cl::init(ExternalCallPolicy::Concrete),
    cl::cat(ExtCallsCat));


/*** External call warnings options ***/

enum class ExtCallWarnings {
  None,            // Never warn on external calls
  OncePerFunction, // Only warn once per function on external calls
  All,             // Always warn on external calls
};

cl::opt<ExtCallWarnings> ExternalCallWarnings(
    "external-call-warnings",
    cl::desc("Specify when to warn about external calls"),
    cl::values(
        clEnumValN(
            ExtCallWarnings::None, "none",
            "Never warn"),
        clEnumValN(ExtCallWarnings::OncePerFunction, "once-per-function",
                   "Warn once per external function (default)"),
        clEnumValN(ExtCallWarnings::All, "all",
                   "Always warn")),
    cl::init(ExtCallWarnings::OncePerFunction),
    cl::cat(ExtCallsCat));

cl::opt<std::size_t> ExternalPageThreshold(
    "kdalloc-external-page-threshold", cl::init(1024),
    cl::desc(
        "Threshold for garbage collecting pages used by external calls. If "
        "there is a significant number of infrequently used pages resident in "
        "memory, these will only be cleaned up if the total number of pages "
        "used for external calls is above the given threshold (default=1024)."),
    cl::cat(ExtCallsCat));

/*** Seeding options ***/

cl::opt<bool> AlwaysOutputSeeds(
    "always-output-seeds",
    cl::init(true),
    cl::desc(
        "Dump test cases even if they are driven by seeds only (default=true)"),
    cl::cat(SeedingCat));

cl::opt<bool> OnlyReplaySeeds(
    "only-replay-seeds",
    cl::init(false),
    cl::desc("Discard states that do not have a seed (default=false)."),
    cl::cat(SeedingCat));

cl::opt<bool> OnlySeed("only-seed",
                       cl::init(false),
                       cl::desc("Stop execution after seeding is done without "
                                "doing regular search (default=false)."),
                       cl::cat(SeedingCat));

cl::opt<bool> AllowSeedExtension(
    "allow-seed-extension", cl::init(false),
    cl::desc("Allow extra values to become symbolic during seeding; "
             "the seed is extended with zeros (default=false)."),
    cl::cat(SeedingCat));

cl::opt<bool> AllowSeedTruncation(
    "allow-seed-truncation",
    cl::init(false),
    cl::desc("Allow smaller buffers than in seeds (default=false)."),
    cl::cat(SeedingCat));

cl::opt<bool> NamedSeedMatching(
    "named-seed-matching",
    cl::init(false),
    cl::desc("Use names to match symbolic objects to inputs (default=false)."),
    cl::cat(SeedingCat));

cl::opt<std::string>
    SeedTime("seed-time",
             cl::desc("Amount of time to dedicate to seeds, before normal "
                      "search (default=0s (off))"),
             cl::cat(SeedingCat));


/*** Termination criteria options ***/

cl::list<StateTerminationType> ExitOnErrorType(
    "exit-on-error-type",
    cl::desc("Stop execution after reaching a specified condition (default=false)"),
    cl::values(
        clEnumValN(StateTerminationType::Abort, "Abort",
                   "The program reached abort or klee_abort"),
        clEnumValN(StateTerminationType::Assert, "Assert",
                   "An assertion was hit"),
        clEnumValN(StateTerminationType::BadVectorAccess, "BadVectorAccess",
                   "Vector accessed out of bounds"),
        clEnumValN(StateTerminationType::Execution, "Execution",
                   "Trying to execute an unexpected instruction"),
        clEnumValN(StateTerminationType::External, "External",
                   "External objects referenced"),
        clEnumValN(StateTerminationType::Free, "Free",
                   "Freeing invalid memory"),
        clEnumValN(StateTerminationType::Model, "Model",
                   "Memory model limit hit"),
        clEnumValN(StateTerminationType::Overflow, "Overflow",
                   "An overflow occurred"),
        clEnumValN(StateTerminationType::Ptr, "Ptr", "Pointer error"),
        clEnumValN(StateTerminationType::ReadOnly, "ReadOnly",
                   "Write to read-only memory"),
        clEnumValN(StateTerminationType::ReportError, "ReportError",
                   "klee_report_error called"),
        clEnumValN(StateTerminationType::InvalidBuiltin, "InvalidBuiltin",
                   "Passing invalid value to compiler builtin"),
        clEnumValN(StateTerminationType::ImplicitTruncation, "ImplicitTruncation",
                   "Implicit conversion from integer of larger bit width to "
                   "smaller bit width that results in data loss"),
        clEnumValN(StateTerminationType::ImplicitConversion, "ImplicitConversion",
                   "Implicit conversion between integer types that changes the "
                   "sign of the value"),
        clEnumValN(StateTerminationType::UnreachableCall, "UnreachableCall",
                   "Control flow reached an unreachable program point"),
        clEnumValN(StateTerminationType::MissingReturn, "MissingReturn",
                   "Reaching the end of a value-returning function without "
                   "returning a value"),
        clEnumValN(StateTerminationType::InvalidLoad, "InvalidLoad",
                   "Load of a value which is not in the range of representable "
                   "values for that type"),
        clEnumValN(StateTerminationType::NullableAttribute, "NullableAttribute",
                   "Violation of nullable attribute detected"),
        clEnumValN(StateTerminationType::User, "User",
                   "Wrong klee_* function invocation")),
    cl::ZeroOrMore,
    cl::cat(TerminationCat));

cl::opt<unsigned long long> MaxInstructions(
    "max-instructions",
    cl::desc("Stop execution after this many instructions.  Set to 0 to disable (default=0)"),
    cl::init(0),
    cl::cat(TerminationCat));

cl::opt<unsigned>
    MaxForks("max-forks",
             cl::desc("Only fork this many times.  Set to -1 to disable (default=-1)"),
             cl::init(~0u),
             cl::cat(TerminationCat));

cl::opt<unsigned> MaxDepth(
    "max-depth",
    cl::desc("Only allow this many symbolic branches.  Set to 0 to disable (default=0)"),
    cl::init(0),
    cl::cat(TerminationCat));

cl::opt<unsigned> MaxMemory("max-memory",
                            cl::desc("Refuse to fork when above this amount of "
                                     "memory (in MB) (see -max-memory-inhibit) and terminate "
                                     "states when additional 100MB allocated (default=10000)"),
                            cl::init(0),
                            cl::cat(TerminationCat));

cl::opt<bool> MaxMemoryInhibit(
    "max-memory-inhibit",
    cl::desc(
        "Inhibit forking when above memory cap (see -max-memory) (default=true)"),
    cl::init(true),
    cl::cat(TerminationCat));

cl::opt<unsigned> RuntimeMaxStackFrames(
    "max-stack-frames",
    cl::desc("Terminate a state after this many stack frames.  Set to 0 to "
             "disable (default=8192)"),
    cl::init(8192),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticForkPct(
    "max-static-fork-pct", cl::init(1.),
    cl::desc("Maximum percentage spent by an instruction forking out of the "
             "forking of all instructions (default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticSolvePct(
    "max-static-solve-pct", cl::init(1.),
    cl::desc("Maximum percentage of solving time that can be spent by a single "
             "instruction over total solving time for all instructions "
             "(default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticCPForkPct(
    "max-static-cpfork-pct", cl::init(1.),
    cl::desc("Maximum percentage spent by an instruction of a call path "
             "forking out of the forking of all instructions in the call path "
             "(default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticCPSolvePct(
    "max-static-cpsolve-pct", cl::init(1.),
    cl::desc("Maximum percentage of solving time that can be spent by a single "
             "instruction of a call path over total solving time for all "
             "instructions (default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<unsigned> MaxStaticPctCheckDelay(
    "max-static-pct-check-delay",
    cl::desc("Number of forks after which the --max-static-*-pct checks are enforced (default=1000)"),
    cl::init(1000),
    cl::cat(TerminationCat));

cl::opt<std::string> TimerInterval(
    "timer-interval",
    cl::desc("Minimum interval to check timers. "
             "Affects -max-time, -istats-write-interval, -stats-write-interval, and -uncovered-update-interval (default=1s)"),
    cl::init("1s"),
    cl::cat(TerminationCat));


/*** Debugging options ***/

/// The different query logging solvers that can switched on/off
enum PrintDebugInstructionsType {
  STDERR_ALL, ///
  STDERR_SRC,
  STDERR_COMPACT,
  FILE_ALL,    ///
  FILE_SRC,    ///
  FILE_COMPACT ///
};

llvm::cl::bits<PrintDebugInstructionsType> DebugPrintInstructions(
    "debug-print-instructions",
    llvm::cl::desc("Log instructions during execution."),
    llvm::cl::values(
        clEnumValN(STDERR_ALL, "all:stderr",
                   "Log all instructions to stderr "
                   "in format [src, inst_id, "
                   "llvm_inst]"),
        clEnumValN(STDERR_SRC, "src:stderr",
                   "Log all instructions to stderr in format [src, inst_id]"),
        clEnumValN(STDERR_COMPACT, "compact:stderr",
                   "Log all instructions to stderr in format [inst_id]"),
        clEnumValN(FILE_ALL, "all:file",
                   "Log all instructions to file "
                   "instructions.txt in format [src, "
                   "inst_id, llvm_inst]"),
        clEnumValN(FILE_SRC, "src:file",
                   "Log all instructions to file "
                   "instructions.txt in format [src, "
                   "inst_id]"),

        clEnumValN(FILE_COMPACT, "compact:file",
                   "Log all instructions to file instructions.txt in format "
                   "[inst_id]")),
    llvm::cl::CommaSeparated,
    cl::cat(DebugCat));

#ifdef HAVE_ZLIB_H
cl::opt<bool> DebugCompressInstructions(
    "debug-compress-instructions", cl::init(false),
    cl::desc(
        "Compress the logged instructions in gzip format (default=false)."),
    cl::cat(DebugCat));
#endif

cl::opt<bool> DebugCheckForImpliedValues(
    "debug-check-for-implied-values", cl::init(false),
    cl::desc("Debug the implied value optimization"),
    cl::cat(DebugCat));

} // namespace

// XXX hack
extern "C" unsigned dumpStates, dumpExecutionTree;
unsigned dumpStates = 0, dumpExecutionTree = 0;
int KLoopInfo::counter = 0;
int Executor::kernelConfigCounter = 0;
static const int largeSymbolSize = 39973;
static const int loopSymMax = 8;

Executor::Executor(LLVMContext &ctx, const InterpreterOptions &opts,
                   InterpreterHandler *ih)
    : Interpreter(opts), interpreterHandler(ih), searcher(0),
      externalDispatcher(new ExternalDispatcher(ctx)), statsTracker(0),
      pathWriter(0), symPathWriter(0), specialFunctionHandler(0), timers{time::Span(TimerInterval)},
      replayKTest(0), replayPath(0), usingSeeds(0),
      atMemoryLimit(false), inhibitForking(false), haltExecution(false),
      ivcEnabled(false), debugLogBuffer(debugBufferString) {


  const time::Span maxTime{MaxTime};
  if (maxTime) timers.add(
        std::make_unique<Timer>(maxTime, [&]{
        klee_message("HaltTimer invoked");
        setHaltExecution(true);
      }));

  coreSolverTimeout = time::Span{MaxCoreSolverTime};
  if (coreSolverTimeout) UseForkedCoreSolver = true;
  std::unique_ptr<Solver> coreSolver = klee::createCoreSolver(CoreSolverToUse);
  if (!coreSolver) {
    klee_error("Failed to create core solver\n");
  }

  std::unique_ptr<Solver> solver = constructSolverChain(
      std::move(coreSolver),
      interpreterHandler->getOutputFilename(ALL_QUERIES_SMT2_FILE_NAME),
      interpreterHandler->getOutputFilename(SOLVER_QUERIES_SMT2_FILE_NAME),
      interpreterHandler->getOutputFilename(ALL_QUERIES_KQUERY_FILE_NAME),
      interpreterHandler->getOutputFilename(SOLVER_QUERIES_KQUERY_FILE_NAME));

  this->solver = std::make_unique<TimingSolver>(std::move(solver), EqualitySubstitution);
  memory = std::make_unique<MemoryManager>(&arrayCache);

  initializeSearchOptions();

  if (OnlyOutputStatesCoveringNew && !StatsTracker::useIStats())
    klee_error("To use --only-output-states-covering-new, you need to enable --output-istats.");

  if (DebugPrintInstructions.isSet(FILE_ALL) ||
      DebugPrintInstructions.isSet(FILE_COMPACT) ||
      DebugPrintInstructions.isSet(FILE_SRC)) {
    std::string debug_file_name =
        interpreterHandler->getOutputFilename("instructions.txt");
    std::string error;
#ifdef HAVE_ZLIB_H
    if (!DebugCompressInstructions) {
#endif
      debugInstFile = klee_open_output_file(debug_file_name, error);
#ifdef HAVE_ZLIB_H
    } else {
      debug_file_name.append(".gz");
      debugInstFile = klee_open_compressed_output_file(debug_file_name, error);
    }
#endif
    if (!debugInstFile) {
      klee_error("Could not open file %s : %s", debug_file_name.c_str(),
                 error.c_str());
    }
  }
}

llvm::Module *
Executor::setModule(std::vector<std::unique_ptr<llvm::Module>> &modules,
                    const ModuleOptions &opts) {
  assert(!kmodule && !modules.empty() &&
         "can only register one module"); // XXX gross

  kmodule = std::unique_ptr<KModule>(new KModule());

  // Preparing the final module happens in multiple stages

  // Link with KLEE intrinsics library before running any optimizations
  SmallString<128> LibPath(opts.LibraryDir);
  llvm::sys::path::append(LibPath,
                          "libkleeRuntimeIntrinsic" + opts.OptSuffix + ".bca");
  std::string error;
  if (!klee::loadFile(LibPath.c_str(), modules[0]->getContext(), modules,
                      error)) {
    klee_error("Could not load KLEE intrinsic file %s", LibPath.c_str());
  }

  while (kmodule->link(modules, opts.EntryPoint)) {
    kmodule->instrument(opts);
  }


  // Create a list of functions that should be preserved if used
  std::vector<const char *> preservedFunctions;
  specialFunctionHandler = new SpecialFunctionHandler(*this);
  specialFunctionHandler->prepare(preservedFunctions);

  preservedFunctions.push_back(opts.EntryPoint.c_str());

  // Preserve the free-standing library calls
  preservedFunctions.push_back("memset");
  preservedFunctions.push_back("memcpy");
  preservedFunctions.push_back("memcmp");
  preservedFunctions.push_back("memmove");

  kmodule->optimiseAndPrepare(opts, preservedFunctions);
  kmodule->checkModule();

  kmodule->manifest(interpreterHandler, StatsTracker::useStatistics());

  specialFunctionHandler->bind();

  if (StatsTracker::useStatistics() || userSearcherRequiresMD2U()) {
    statsTracker = 
      new StatsTracker(*this,
                       interpreterHandler->getOutputFilename("assembly.ll"),
                       userSearcherRequiresMD2U());
  }

  // Initialize the context.
  DataLayout *TD = kmodule->targetData.get();

  Context::initialize(TD->isLittleEndian(),
                    (Expr::Width)TD->getPointerSizeInBits());

  return kmodule->module.get();
}

Executor::~Executor() {
  delete externalDispatcher;
  delete specialFunctionHandler;
  delete statsTracker;
}


void Executor::saveLoopInfo(const llvm::Loop *L, ExecutionState &state) {
  if (!L) return;
  unsigned numBlocks = L->getNumBlocks();

  llvm::BasicBlock *header = L->getHeader();
  if (header) {
    if (!header->empty()) {
        llvm::Instruction *lastInst = header->getTerminator(); // The last instruction
        if (llvm::BranchInst *headerBranch = llvm::dyn_cast<llvm::BranchInst>(lastInst)) {
          std::set<llvm::BasicBlock*> bodyBlocks(L->block_begin(), L->block_end());

          if (headerBranch->isUnconditional()) {
            state.loopInfoMap[L->getLoopLatch()] = new KLoopInfo(header, false, numBlocks == 2, std::move(bodyBlocks));
            // if (!state.loopInfoMap[L->getLoopLatch()]->backedge) {
            //   llvm::outs() << "loop backage null\n";
            // }
          } else {
            // continue block can also be latches: pick the last latch block
            SmallVector<llvm::BasicBlock*, 4> latches;
            L->getLoopLatches(latches);
            llvm::BasicBlock *lastLatch = nullptr;
            if (!latches.empty()) {
                lastLatch = latches.back();  // Pick the "last" latch
            }

            state.loopInfoMap[header] = new KLoopInfo(lastLatch, true, numBlocks == 2, std::move(bodyBlocks));
            if (lastLatch && !lastLatch->empty() && lastLatch->size() == 1) {
              if (auto *br = llvm::dyn_cast<llvm::BranchInst>(lastLatch->getTerminator())) {
                if (br->isUnconditional()) {
                  state.loopInfoMap[header]->backedge = header;
                }
              }
            }
          }
        } else {
          llvm::BasicBlock *succ = header->getTerminator()->getSuccessor(0);
          if (succ && !succ->empty()) {
            if (auto *succBr = llvm::dyn_cast<llvm::BranchInst>(succ->getTerminator())) {
              if (!succBr->isUnconditional()) {
                std::set<llvm::BasicBlock*> bodyBlocks(L->block_begin(), L->block_end());
                state.loopInfoMap[succ] =
                    new KLoopInfo(L->getLoopLatch(), true, numBlocks == 2, std::move(bodyBlocks));
              }
            }
          }
        }
    }
  }

  for (const llvm::Loop *subLoop : L->getSubLoops()) {
      saveLoopInfo(subLoop, state);
    }
}

bool Executor::isMemFuncName(const std::string &funcName) {
  return funcName == "memcpy" || funcName == "memset" || funcName == "memcmp" || funcName == "memmove";
}

void Executor::initializeLoopInfo(llvm::Module *M, ExecutionState &state) {
  for (Function &F : *M) {
    if (F.getIntrinsicID()!=Intrinsic::not_intrinsic || isMemFuncName(F.getName().str())) {
      continue;
    }

    if (F.getName().find("detail14__to_chars_len")!=std::string::npos) {
      continue;
    }

    if (!F.isDeclaration()) {
      llvm::DominatorTree DT(F);
      llvm::LoopInfo LI(DT);

      for (llvm::Loop *L : LI) {
          saveLoopInfo(L, state);
      }
    }
  }
}

void Executor::findTrampolineBranches(llvm::Module *M) {
  for (Function &F : *M) {
    if (F.isDeclaration()) continue; // skip external functions

    if (F.getIntrinsicID()!=Intrinsic::not_intrinsic || isMemFuncName(F.getName().str())) {
      continue;
    }

    if (F.getName().find("detail14__to_chars_len")!=std::string::npos) {
      continue;
    }

    for (BasicBlock &BB : F) {
      Instruction *term = BB.getTerminator();
      if (auto *br = dyn_cast<BranchInst>(term)) {
        // only care about conditional branches
        if (!br->isConditional()) continue;

        BasicBlock *B = br->getSuccessor(0);
        BasicBlock *C = br->getSuccessor(1);
        BasicBlock *D = nullptr;
        if (C && !C->empty() && C->size() == 1) {
          if (auto *br_C = llvm::dyn_cast<llvm::BranchInst>(C->getTerminator())) {
            if (br_C->isUnconditional()) {
              D = br_C->getSuccessor(0);
            }
          }
        }

        // check if block B unconditionally branches to C
        if (BranchInst *brB = dyn_cast<BranchInst>(B->getTerminator())) {
          if (brB->isUnconditional() && brB->getSuccessor(0) == C) {
            // Get the first instruction of C
            if (C->empty()) continue;
            llvm::Instruction *firstInstC = &C->front();
            if (firstInstC->getOpcode() == Instruction::Ret) continue;
            trampolineBrs[br] = firstInstC;
          } else if (brB->isUnconditional() && brB->getSuccessor(0) == D) {
            if (C->empty()) continue;
            llvm::Instruction *firstInstD = &D->front();
            if (firstInstD->getOpcode() == Instruction::Ret) continue;
            trampolineBrs[br] = firstInstD;
          }
        }
      }
    }
  }
}

/***/

void Executor::initializeGlobalObject(ExecutionState &state, ObjectState *os,
                                      const Constant *c, 
                                      unsigned offset) {
  const auto targetData = kmodule->targetData.get();
  if (const ConstantVector *cp = dyn_cast<ConstantVector>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cp->getType()->getElementType());
    for (unsigned i=0, e=cp->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cp->getOperand(i), 
			     offset + i*elementSize);
  } else if (isa<ConstantAggregateZero>(c)) {
    unsigned i, size = targetData->getTypeStoreSize(c->getType());
    for (i=0; i<size; i++)
      os->write8(offset+i, (uint8_t) 0);
  } else if (const ConstantArray *ca = dyn_cast<ConstantArray>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(ca->getType()->getElementType());
    for (unsigned i=0, e=ca->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, ca->getOperand(i), 
			     offset + i*elementSize);
  } else if (const ConstantStruct *cs = dyn_cast<ConstantStruct>(c)) {
    const StructLayout *sl =
      targetData->getStructLayout(cast<StructType>(cs->getType()));
    for (unsigned i=0, e=cs->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cs->getOperand(i), 
			     offset + sl->getElementOffset(i));
  } else if (const ConstantDataSequential *cds =
               dyn_cast<ConstantDataSequential>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cds->getElementType());
    for (unsigned i=0, e=cds->getNumElements(); i != e; ++i)
      initializeGlobalObject(state, os, cds->getElementAsConstant(i),
                             offset + i*elementSize);
  } else if (!isa<UndefValue>(c) && !isa<MetadataAsValue>(c)) {
    unsigned StoreBits = targetData->getTypeStoreSizeInBits(c->getType());
    ref<ConstantExpr> C = evalConstant(c);

    assert(StoreBits >= C->getWidth() && "Invalid store size!");
    if (StoreBits > C->getWidth())
      C = C->ZExt(StoreBits);

    os->write(offset, C);
  }
}

MemoryObject * Executor::addExternalObject(ExecutionState &state, 
                                           void *addr, unsigned size, 
                                           bool isReadOnly) {
  auto mo = memory->allocateFixed(reinterpret_cast<std::uint64_t>(addr),
                                  size, nullptr);
  ObjectState *os = bindObjectInState(state, mo, false);
  for(unsigned i = 0; i < size; i++)
    os->write8(i, ((uint8_t*)addr)[i]);
  if(isReadOnly)
    os->setReadOnly(true);  
  return mo;
}


extern void *__dso_handle __attribute__ ((__weak__));

void Executor::initializeGlobals(ExecutionState &state) {
  // allocate and initialize globals, done in two passes since we may
  // need address of a global in order to initialize some other one.

  // allocate memory objects for all globals
  allocateGlobalObjects(state);

  // initialize aliases first, may be needed for global objects
  initializeGlobalAliases();

  // finally, do the actual initialization
  initializeGlobalObjects(state);
}

void Executor::allocateGlobalObjects(ExecutionState &state) {
  Module *m = kmodule->module.get();

  if (m->getModuleInlineAsm() != "")
    klee_warning("executable has module level assembly (ignoring)");
  // represent function globals using the address of the actual llvm function
  // object. given that we use malloc to allocate memory in states this also
  // ensures that we won't conflict. we don't need to allocate a memory object
  // since reading/writing via a function pointer is unsupported anyway.
  for (Function &f : *m) {
    ref<ConstantExpr> addr;

    // If the symbol has external weak linkage then it is implicitly
    // should be null.
    if (f.hasExternalWeakLinkage() &&
        !externalDispatcher->resolveSymbol(f.getName().str())) {
      addr = Expr::createPointer(0);
    } else {
      // We allocate an object to represent each function,
      // its address can be used for function pointers.
      // TODO: Check whether the object is accessed?
      auto mo = memory->allocate(8, false, true, &state, &f, 8);
      addr = Expr::createPointer(mo->address);
      legalFunctions.emplace(mo->address, &f);
    }

    globalAddresses.emplace(&f, addr);
  }

#ifndef WINDOWS
  int *errno_addr = getErrnoLocation(state);
  MemoryObject *errnoObj =
      addExternalObject(state, (void *)errno_addr, sizeof *errno_addr, false);
  // Copy values from and to program space explicitly
  errnoObj->isUserSpecified = true;
#endif

  // Disabled, we don't want to promote use of live externals.
#ifdef HAVE_CTYPE_EXTERNALS
#ifndef WINDOWS
#ifndef DARWIN
  /* from /usr/include/ctype.h:
       These point into arrays of 384, so they can be indexed by any `unsigned
       char' value [0,255]; by EOF (-1); or by any `signed char' value
       [-128,-1).  ISO C requires that the ctype functions work for `unsigned */
  const uint16_t **addr = __ctype_b_loc();
  addExternalObject(state, const_cast<uint16_t*>(*addr-128),
                    384 * sizeof **addr, true);
  addExternalObject(state, addr, sizeof(*addr), true);
    
  const int32_t **lower_addr = __ctype_tolower_loc();
  addExternalObject(state, const_cast<int32_t*>(*lower_addr-128),
                    384 * sizeof **lower_addr, true);
  addExternalObject(state, lower_addr, sizeof(*lower_addr), true);
  
  const int32_t **upper_addr = __ctype_toupper_loc();
  addExternalObject(state, const_cast<int32_t*>(*upper_addr-128),
                    384 * sizeof **upper_addr, true);
  addExternalObject(state, upper_addr, sizeof(*upper_addr), true);
#endif
#endif
#endif

  for (const GlobalVariable &v : m->globals()) {
    std::size_t globalObjectAlignment = getAllocationAlignment(&v);
    Type *ty = v.getValueType();
    std::uint64_t size = 0;
    if (ty->isSized())
      size = kmodule->targetData->getTypeStoreSize(ty);
    
    bool isDynamicShared = false;

    if (v.isDeclaration()) {
      // FIXME: We have no general way of handling unknown external
      // symbols. If we really cared about making external stuff work
      // better we could support user definition, or use the EXE style
      // hack where we check the object file information.

      if (!ty->isSized()) {
        klee_warning("Type for %.*s is not sized",
                     static_cast<int>(v.getName().size()), v.getName().data());
      }

      // XXX - DWD - hardcode some things until we decide how to fix.
#ifndef WINDOWS
      if (v.getName() == "_ZTVN10__cxxabiv117__class_type_infoE") {
        size = 0x2C;
      } else if (v.getName() == "_ZTVN10__cxxabiv120__si_class_type_infoE") {
        size = 0x2C;
      } else if (v.getName() == "_ZTVN10__cxxabiv121__vmi_class_type_infoE") {
        size = 0x2C;
      }
#endif

      if (size == 0) {
        if (v.getLinkage() == GlobalValue::ExternalLinkage && v.getType()->getPointerAddressSpace() == 3) {
          isDynamicShared = true;
          size = 96*1024;
        } else {
          klee_warning("Unable to find size for global variable: %.*s (use will "
                     "result in out of bounds access)",
                     static_cast<int>(v.getName().size()), v.getName().data());
        }
      }
    }

    MemoryObject *mo;

    if (v.getType()->getPointerAddressSpace() == 3) {
      mo = memory->allocateShared(size, /*state=*/nullptr, /*allocSite=*/&v);
      mo->memType= MemoryObject::MemType::SHARED;
      sharedAddresses[mo->getBaseExpr()] = std::make_pair(v.getName().str(), isDynamicShared);
    } else {
      mo = memory->allocate(size, /*isLocal=*/false,
                                        /*isGlobal=*/true, /*state=*/nullptr,
                                        /*allocSite=*/&v,
                                        /*alignment=*/globalObjectAlignment);
      mo->memType = MemoryObject::MemType::GLOBAL;
    }

    if (!mo)
      klee_error("out of memory");
    globalObjects.emplace(&v, mo);
    globalAddresses.emplace(&v, mo->getBaseExpr());
  }
}

void Executor::initializeGlobalAlias(const llvm::Constant *c) {
  // aliasee may either be a global value or constant expression
  const auto *ga = dyn_cast<GlobalAlias>(c);
  if (ga) {
    if (globalAddresses.count(ga)) {
      // already resolved by previous invocation
      return;
    }
    const llvm::Constant *aliasee = ga->getAliasee();
    if (const auto *gv = dyn_cast<GlobalValue>(aliasee)) {
      // aliasee is global value
      auto it = globalAddresses.find(gv);
      // uninitialized only if aliasee is another global alias
      if (it != globalAddresses.end()) {
        globalAddresses.emplace(ga, it->second);
        return;
      }
    }
  }

  // resolve aliases in all sub-expressions
  for (const auto *op : c->operand_values()) {
    initializeGlobalAlias(cast<Constant>(op));
  }

  if (ga) {
    globalAddresses.emplace(ga, evalConstant(ga->getAliasee()));
  }
}

void Executor::initializeGlobalAliases() {
  const Module *m = kmodule->module.get();
  for (const GlobalAlias &a : m->aliases()) {
    initializeGlobalAlias(&a);
  }
}

void Executor::initializeGlobalObjects(ExecutionState &state) {
  const Module *m = kmodule->module.get();

  for (const GlobalVariable &v : m->globals()) {
    MemoryObject *mo = globalObjects.find(&v)->second;
    ObjectState *os;
    if (sharedAddresses.find(mo->getBaseExpr()) != sharedAddresses.end()) {
      bool isDynamic = sharedAddresses[mo->getBaseExpr()].second;
      if(!isDynamic) {
        std::string name = sharedAddresses[mo->getBaseExpr()].first;
        const Array *array = arrayCache.CreateArray(name, mo->size);
        mo->setName(name);
        os = bindObjectInState(state, mo, false, array);
        state.addSymbolic(mo, array);

        SymArrayMemoryObject *symmo = new SymArrayMemoryObject(mo->address, array, nullptr, ConstantExpr::create(mo->size, Expr::Int64));
        state.symbolicArrayMap[name] = symmo;

        const llvm::Type *valTy = v.getValueType();
        if (const llvm::ArrayType *arrTy = llvm::dyn_cast<llvm::ArrayType>(valTy)) {
          const llvm::Type *elemTy = arrTy->getElementType();
          if (elemTy->isIntegerTy()) {
            symmo->isFloat = false;
          }
          if (elemTy->isFloatingPointTy()) {
            symmo->isFloat = true;
          }
        }
      } else {
        os = bindObjectInState(state, mo, false);
      }
    } else {
      os = bindObjectInState(state, mo, false);
    }

    if (v.isDeclaration() && mo->size && !(v.getLinkage() == GlobalValue::ExternalLinkage && v.getType()->getPointerAddressSpace() == 3)) {
      // Read concrete value and write it to our copy.
      void *addr;
      if (v.getName() == "__dso_handle") {
        addr = &__dso_handle; // wtf ?
      } else {
        addr = externalDispatcher->resolveSymbol(v.getName().str());
      }
      if (!addr) {
        os->initializeToZero();
      } else {
        for (unsigned offset = 0; offset < mo->size; offset++) {
          os->write8(offset, static_cast<unsigned char *>(addr)[offset]);
        }
      }
    } else if (v.hasInitializer()) {
      initializeGlobalObject(state, os, v.getInitializer(), 0);
      if (v.isConstant()) {
        os->setReadOnly(true);
        // initialise constant memory that may be used with external calls
        state.addressSpace.copyOutConcrete(mo, os);
      }
    } else {
      os->initializeToZero();
    }
  }
}


bool Executor::branchingPermitted(const ExecutionState &state) const {
  if ((MaxMemoryInhibit && atMemoryLimit) ||
      state.forkDisabled ||
      inhibitForking ||
      (MaxForks!=~0u && stats::forks >= MaxForks)) {

    if (MaxMemoryInhibit && atMemoryLimit)
      klee_warning_once(0, "skipping fork (memory cap exceeded)");
    else if (state.forkDisabled)
      klee_warning_once(0, "skipping fork (fork disabled on current path)");
    else if (inhibitForking)
      klee_warning_once(0, "skipping fork (fork disabled globally)");
    else
      klee_warning_once(0, "skipping fork (max-forks reached)");

    return false;
  }

  return true;
}

void Executor::branch(ExecutionState &state,
                      const std::vector<ref<Expr>> &conditions,
                      std::vector<ExecutionState *> &result,
                      BranchType reason) {
  TimerStatIncrementer timer(stats::forkTime);
  unsigned N = conditions.size();
  assert(N);

  if (!branchingPermitted(state)) {
    unsigned next = theRNG.getInt32() % N;
    for (unsigned i=0; i<N; ++i) {
      if (i == next) {
        result.push_back(&state);
      } else {
        result.push_back(nullptr);
      }
    }
    stats::inhibitedForks += N - 1;
  } else {
    stats::forks += N-1;
    stats::incBranchStat(reason, N-1);

    // XXX do proper balance or keep random?
    result.push_back(&state);
    for (unsigned i=1; i<N; ++i) {
      ExecutionState *es = result[theRNG.getInt32() % i];
      ExecutionState *ns = es->branch();
      addedStates.push_back(ns);
      result.push_back(ns);
      executionTree->attach(es->executionTreeNode, ns, es, reason);
    }
  }

  // If necessary redistribute seeds to match conditions, killing
  
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    std::vector<SeedInfo> seeds = it->second;
    seedMap.erase(it);

    // when conditions are mutually exclusive and their conjunction is
    for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
           siie = seeds.end(); siit != siie; ++siit) {
      unsigned i;
      for (i=0; i<N; ++i) {
        ref<ConstantExpr> res;
        bool success = solver->getValue(
            state.constraints, siit->assignment.evaluate(conditions[i]), res,
            state.queryMetaData);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue())
          break;
      }
      
      // If we didn't find a satisfying condition randomly pick one
      if (i==N)
        i = theRNG.getInt32() % N;

      // Extra check in case we're replaying seeds with a max-fork
      if (result[i])
        seedMap[result[i]].push_back(*siit);
    }

    if (OnlyReplaySeeds) {
      for (unsigned i=0; i<N; ++i) {
        if (result[i] && !seedMap.count(result[i])) {
          terminateStateEarlyAlgorithm(*result[i], "Unseeded path during replay", StateTerminationType::Replay);
          result[i] = nullptr;
        }
      }
    }
  }

  for (unsigned i=0; i<N; ++i)
    if (result[i])
      addConstraint(*result[i], conditions[i]);
}

ref<Expr> Executor::maxStaticPctChecks(ExecutionState &current,
                                       ref<Expr> condition) {
  if (isa<klee::ConstantExpr>(condition))
    return condition;

  if (MaxStaticForkPct == 1. && MaxStaticSolvePct == 1. &&
      MaxStaticCPForkPct == 1. && MaxStaticCPSolvePct == 1.)
    return condition;

  // These checks are performed only after at least MaxStaticPctCheckDelay forks
  // have been performed since execution started
  if (stats::forks < MaxStaticPctCheckDelay)
    return condition;

  StatisticManager &sm = *theStatisticManager;
  CallPathNode *cpn = current.stack.back().callPathNode;

  bool reached_max_fork_limit =
      (MaxStaticForkPct < 1. &&
       (sm.getIndexedValue(stats::forks, sm.getIndex()) >
        stats::forks * MaxStaticForkPct));

  bool reached_max_cp_fork_limit = (MaxStaticCPForkPct < 1. && cpn &&
                                    (cpn->statistics.getValue(stats::forks) >
                                     stats::forks * MaxStaticCPForkPct));

  bool reached_max_solver_limit =
      (MaxStaticSolvePct < 1 &&
       (sm.getIndexedValue(stats::solverTime, sm.getIndex()) >
        stats::solverTime * MaxStaticSolvePct));

  bool reached_max_cp_solver_limit =
      (MaxStaticCPForkPct < 1. && cpn &&
       (cpn->statistics.getValue(stats::solverTime) >
        stats::solverTime * MaxStaticCPSolvePct));

  if (reached_max_fork_limit || reached_max_cp_fork_limit ||
      reached_max_solver_limit || reached_max_cp_solver_limit) {
    ref<klee::ConstantExpr> value;
    bool success = solver->getValue(current.constraints, condition, value,
                                    current.queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
    (void)success;

    std::string msg("skipping fork and concretizing condition (MaxStatic*Pct "
                    "limit reached) at ");
    llvm::raw_string_ostream os(msg);
    os << current.prevPC->getSourceLocation();
    klee_warning_once(0, "%s", os.str().c_str());

    addConstraint(current, EqExpr::create(value, condition));
    condition = value;
  }
  return condition;
}

bool Executor::isReadSymArrayWithSymIndex(ExecutionState &current, ref<Expr> expr) {
  if (auto readExpr = dyn_cast<ReadExpr>(expr)) {
    if (readExpr->updates.root && !readExpr->updates.root->name.empty()) {
      std::string name = readExpr->updates.root->name;
      return current.symbolicArrayMap.find(name) != current.symbolicArrayMap.end() && !isa<ConstantExpr>(readExpr->index);
    }
    return false;
  }

  for (unsigned i = 0; i < expr->getNumKids(); i++) {
    if (isReadSymArrayWithSymIndex(current, expr->getKid(i))) {
      return true;
    }
  }

  return false;
}

Executor::StatePair Executor::fork(ExecutionState &current, ref<Expr> condition,
                                   bool isInternal, BranchType reason) {  
  Solver::Validity res;
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&current);
  bool isSeeding = it != seedMap.end();

  if (!isSeeding)
    condition = maxStaticPctChecks(current, condition);

  time::Span timeout = coreSolverTimeout;
  if (isSeeding)
    timeout *= static_cast<unsigned>(it->second.size());
  solver->setTimeout(timeout);
  bool success = solver->evaluate(current.constraints, condition, res,
                                  current.queryMetaData, current.intArrNames, current.getIOExprs());
  solver->setTimeout(time::Span());
  if (!success) {
    current.pc = current.prevPC;
    terminateStateOnSolverError(current, "Query timed out (fork).");
    return StatePair(nullptr, nullptr);
  }

  if (res==Solver::Error) {
    terminateStateEarly(current, "query error", StateTerminationType::ReportError);
    return StatePair(nullptr, nullptr);
  }

  if (!isSeeding) {
    if (replayPath && !isInternal) {
      assert(replayPosition<replayPath->size() &&
             "ran out of branches in replay path mode");
      bool branch = (*replayPath)[replayPosition++];
      
      if (res==Solver::True) {
        assert(branch && "hit invalid branch in replay path mode");
      } else if (res==Solver::False) {
        assert(!branch && "hit invalid branch in replay path mode");
      } else {
        // add constraints
        if(branch) {
          res = Solver::True;
          addConstraint(current, condition);
        } else  {
          res = Solver::False;
          addConstraint(current, Expr::createIsZero(condition));
        }
      }
    } else if (res==Solver::Unknown) {
      assert(!replayKTest && "in replay mode, only one branch can be true.");
      
      if (!branchingPermitted(current)) {
        TimerStatIncrementer timer(stats::forkTime);
        if (theRNG.getBool()) {
          addConstraint(current, condition);
          res = Solver::True;        
        } else {
          addConstraint(current, Expr::createIsZero(condition));
          res = Solver::False;
        }
        ++stats::inhibitedForks;
      }
    }
  }

  // Fix branch in only-replay-seed mode, if we don't have both true
  // and false seeds.
  if (isSeeding && 
      (current.forkDisabled || OnlyReplaySeeds) && 
      res == Solver::Unknown) {
    bool trueSeed=false, falseSeed=false;
    // Is seed extension still ok here?
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      ref<ConstantExpr> res;
      bool success = solver->getValue(current.constraints,
                                      siit->assignment.evaluate(condition), res,
                                      current.queryMetaData);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res->isTrue()) {
        trueSeed = true;
      } else {
        falseSeed = true;
      }
      if (trueSeed && falseSeed)
        break;
    }
    if (!(trueSeed && falseSeed)) {
      assert(trueSeed || falseSeed);
      
      res = trueSeed ? Solver::True : Solver::False;
      addConstraint(current, trueSeed ? condition : Expr::createIsZero(condition));
    }
  }


  // XXX - even if the constraint is provable one way or the other we
  // can probably benefit by adding this constraint and allowing it to
  // reduce the other constraints. For example, if we do a binary
  // search on a particular value, and then see a comparison against
  // the value it has been fixed at, we should take this as a nice
  // hint to just use the single constraint instead of all the binary
  // search ones. If that makes sense.
  if (res==Solver::True) {
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "1";
      }
    }

    return StatePair(&current, nullptr);
  } else if (res==Solver::False) {
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "0";
      }
    }

    return StatePair(nullptr, &current);
  } else {
    TimerStatIncrementer timer(stats::forkTime);
    ExecutionState *falseState, *trueState = &current;

    ++stats::forks;

    falseState = trueState->branch();
    addedStates.push_back(falseState);

    if (it != seedMap.end()) {
      std::vector<SeedInfo> seeds = it->second;
      it->second.clear();
      std::vector<SeedInfo> &trueSeeds = seedMap[trueState];
      std::vector<SeedInfo> &falseSeeds = seedMap[falseState];
      for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
             siie = seeds.end(); siit != siie; ++siit) {
        ref<ConstantExpr> res;
        bool success = solver->getValue(current.constraints,
                                        siit->assignment.evaluate(condition),
                                        res, current.queryMetaData);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue()) {
          trueSeeds.push_back(*siit);
        } else {
          falseSeeds.push_back(*siit);
        }
      }
      
      bool swapInfo = false;
      if (trueSeeds.empty()) {
        if (&current == trueState) swapInfo = true;
        seedMap.erase(trueState);
      }
      if (falseSeeds.empty()) {
        if (&current == falseState) swapInfo = true;
        seedMap.erase(falseState);
      }
      if (swapInfo) {
        std::swap(trueState->coveredNew, falseState->coveredNew);
        std::swap(trueState->coveredLines, falseState->coveredLines);
      }
    }

    executionTree->attach(current.executionTreeNode, falseState, trueState, reason);
    stats::incBranchStat(reason, 1);

    if (pathWriter) {
      // Need to update the pathOS.id field of falseState, otherwise the same id
      // is used for both falseState and trueState.
      falseState->pathOS = pathWriter->open(current.pathOS);
      if (!isInternal) {
        trueState->pathOS << "1";
        falseState->pathOS << "0";
      }
    }
    if (symPathWriter) {
      falseState->symPathOS = symPathWriter->open(current.symPathOS);
      if (!isInternal) {
        trueState->symPathOS << "1";
        falseState->symPathOS << "0";
      }
    }

    addConstraint(*trueState, condition);
    addConstraint(*falseState, Expr::createIsZero(condition));

    // Kinda gross, do we even really still want this option?
    if (MaxDepth && MaxDepth<=trueState->depth) {
      klee_warning("max-depth exceeded.");
      terminateStateEarly(*trueState, "max-depth exceeded.", StateTerminationType::MaxDepth);
      terminateStateEarly(*falseState, "max-depth exceeded.", StateTerminationType::MaxDepth);
      return StatePair(nullptr, nullptr);
    }

    return StatePair(trueState, falseState);
  }
}

void Executor::addConstraint(ExecutionState &state, ref<Expr> condition) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(condition)) {
    if (!CE->isTrue())
      llvm::report_fatal_error("attempt to add invalid constraint");
    return;
  }

  // Check to see if this constraint violates seeds.
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    bool warn = false;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      bool res;
      bool success = solver->mustBeFalse(state.constraints,
                                         siit->assignment.evaluate(condition),
                                         res, state.queryMetaData, state.intArrNames, state.getIOExprs());
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        siit->patchSeed(state, condition, solver.get());
        warn = true;
      }
    }
    if (warn)
      klee_warning("seeds patched for violating constraint"); 
  }

  state.addConstraint(condition);
  if (ivcEnabled)
    doImpliedValueConcretization(state, condition, 
                                 ConstantExpr::alloc(1, Expr::Bool));
}

const Cell& Executor::eval(KInstruction *ki, unsigned index, 
                           ExecutionState &state) const {
  assert(index < ki->inst->getNumOperands());
  int vnumber = ki->operands[index];

  assert(vnumber != -1 &&
         "Invalid operand to eval(), not a value or constant!");

  // Determine if this is a constant or not.
  if (vnumber < 0) {
    unsigned index = -vnumber - 2;
    return kmodule->constantTable[index];
  } else {
    unsigned index = vnumber;
    StackFrame &sf = state.stack.back();
    return sf.locals[index];
  }
}

void Executor::bindLocal(KInstruction *target, ExecutionState &state, 
                         ref<Expr> value) {
  getDestCell(state, target).value = value;
}

void Executor::bindArgument(KFunction *kf, unsigned index, 
                            ExecutionState &state, ref<Expr> value) {
  getArgumentCell(state, kf, index).value = value;
}

ref<Expr> Executor::toUnique(const ExecutionState &state, 
                             ref<Expr> &e) {
  ref<Expr> result = e;

  if (!isa<ConstantExpr>(e)) {
    ref<ConstantExpr> value;
    bool isTrue = false;
    e = optimizer.optimizeExpr(e, true);
    solver->setTimeout(coreSolverTimeout);
    if (solver->getValue(state.constraints, e, value, state.queryMetaData)) {
      ref<Expr> cond = EqExpr::create(e, value);
      cond = optimizer.optimizeExpr(cond, false);
      if (solver->mustBeTrue(state.constraints, cond, isTrue,
                             state.queryMetaData, state.intArrNames, state.getIOExprs()) &&
          isTrue)
        result = value;
    }
    solver->setTimeout(time::Span());
  }
  
  return result;
}

ref<klee::ConstantExpr> Executor::toConstant(ExecutionState &state, ref<Expr> e,
                                             const std::string &reason,
                                             bool concretize) {
  e = ConstraintManager::simplifyExpr(state.constraints, e);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(e))
    return CE;

  /* If no seed evaluation results in a constant, call the solver */
  ref<ConstantExpr> cvalue = getValueFromSeeds(state, e);
  if (!cvalue) {
    [[maybe_unused]] bool success =
        solver->getValue(state.constraints, e, cvalue, state.queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
  }

  std::string str;
  llvm::raw_string_ostream os(str);
  os << "silently concretizing (reason: " << reason << ") expression " << e
     << " to value " << cvalue << " (" << (*(state.pc)).info->file << ":"
     << (*(state.pc)).info->line << " " << (*(state.pc)).info->assemblyLine << ")";

  if (ExternalCallWarnings == ExtCallWarnings::All)
    klee_warning("%s", os.str().c_str());
  else
    klee_warning_once(reason.c_str(), "%s", os.str().c_str());

  if (concretize)
    addConstraint(state, EqExpr::create(e, cvalue));

  return cvalue;
}

ref<klee::ConstantExpr> Executor::getValueFromSeeds(ExecutionState &state,
                                                    ref<Expr> e) {
  auto found = seedMap.find(&state);
  if (found == seedMap.end())
    return nullptr;

  auto seeds = found->second;
  for (auto const &seed : seeds) {
    auto value = seed.assignment.evaluate(e);
    if (isa<ConstantExpr>(value))
      return value;
  }
  return nullptr;
}

void Executor::executeGetValue(ExecutionState &state,
                               ref<Expr> e,
                               KInstruction *target) {
  e = ConstraintManager::simplifyExpr(state.constraints, e);
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it==seedMap.end() || isa<ConstantExpr>(e)) {
    ref<ConstantExpr> value;
    e = optimizer.optimizeExpr(e, true);
    bool success =
        solver->getValue(state.constraints, e, value, state.queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    bindLocal(target, state, value);
  } else {
    std::set< ref<Expr> > values;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      ref<Expr> cond = siit->assignment.evaluate(e);
      cond = optimizer.optimizeExpr(cond, true);
      ref<ConstantExpr> value;
      bool success =
          solver->getValue(state.constraints, cond, value, state.queryMetaData);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      values.insert(value);
    }
    
    std::vector< ref<Expr> > conditions;
    for (std::set< ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit)
      conditions.push_back(EqExpr::create(e, *vit));

    std::vector<ExecutionState*> branches;
    branch(state, conditions, branches, BranchType::GetVal);
    
    std::vector<ExecutionState*>::iterator bit = branches.begin();
    for (std::set< ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit) {
      ExecutionState *es = *bit;
      if (es)
        bindLocal(target, *es, *vit);
      ++bit;
    }
  }
}

void Executor::printDebugInstructions(ExecutionState &state) {
  // print nothing if option unset
  if (DebugPrintInstructions.getBits() == 0)
    return;

  llvm::raw_ostream *stream = nullptr;
  if (DebugPrintInstructions.isSet(STDERR_ALL) ||
      DebugPrintInstructions.isSet(STDERR_SRC) ||
      DebugPrintInstructions.isSet(STDERR_COMPACT))
    stream = &llvm::errs();
  else
    stream = &debugLogBuffer;

  // print:
  //   [all]     src location:asm line:state ID:instruction
  //   [compact]              asm line:state ID
  //   [src]     src location:asm line:state ID
  if (!DebugPrintInstructions.isSet(STDERR_COMPACT) &&
      !DebugPrintInstructions.isSet(FILE_COMPACT)) {
    (*stream) << "     " << state.pc->getSourceLocation() << ':';
  }

  (*stream) << state.pc->info->assemblyLine << ':' << state.getID();

  if (DebugPrintInstructions.isSet(STDERR_ALL) ||
      DebugPrintInstructions.isSet(FILE_ALL))
    (*stream) << ':' << *(state.pc->inst);

  (*stream) << '\n';

  // flush to file
  if (DebugPrintInstructions.isSet(FILE_ALL) ||
      DebugPrintInstructions.isSet(FILE_COMPACT) ||
      DebugPrintInstructions.isSet(FILE_SRC)) {
    debugLogBuffer.flush();
    (*debugInstFile) << debugLogBuffer.str();
    debugBufferString = "";
  }
}

void Executor::stepInstruction(ExecutionState &state) {
  printDebugInstructions(state);
  if (statsTracker)
    statsTracker->stepInstruction(state);

  ++stats::instructions;
  ++state.steppedInstructions;
  state.prevPC = state.pc;
  ++state.pc;

  if (stats::instructions == MaxInstructions)
    haltExecution = true;
}

static inline const llvm::fltSemantics *fpWidthToSemantics(unsigned width) {
  switch (width) {
  case Expr::Int32:
    return &llvm::APFloat::IEEEsingle();
  case Expr::Int64:
    return &llvm::APFloat::IEEEdouble();
  case Expr::Fl80:
    return &llvm::APFloat::x87DoubleExtended();
  default:
    return 0;
  }
}

MemoryObject *Executor::serializeLandingpad(ExecutionState &state,
                                            const llvm::LandingPadInst &lpi,
                                            bool &stateTerminated) {
  stateTerminated = false;

  std::vector<unsigned char> serialized;
  for (unsigned current_clause_id = 0; current_clause_id < lpi.getNumClauses();
       ++current_clause_id) {
    if (lpi.isCatch(current_clause_id)) {
      // catch-clause
      serialized.push_back(0);

      std::uint64_t ti_addr = 0;

      llvm::Constant *catchClause = lpi.getClause(current_clause_id);
      llvm::Constant *typeInfo = catchClause->stripPointerCasts();
      if (auto *gv = dyn_cast<llvm::GlobalVariable>(typeInfo)) {
        ti_addr = globalAddresses[gv]->getZExtValue();
      } else if (auto *clause_bitcast =
                     dyn_cast<llvm::BitCastOperator>(catchClause)) {
        auto *clause_type =
            dyn_cast<GlobalValue>(clause_bitcast->getOperand(0));

        ti_addr = globalAddresses[clause_type]->getZExtValue();
      } else if (catchClause->isNullValue()) {
        ti_addr = 0;
      } else {
        terminateStateOnExecError(
            state, "Internal: Clause is not a bitcast or null (catch-all)");
        stateTerminated = true;
        return nullptr;
      }
      const std::size_t old_size = serialized.size();
      serialized.resize(old_size + 8);
      memcpy(serialized.data() + old_size, &ti_addr, sizeof(ti_addr));
    } else if (lpi.isFilter(current_clause_id)) {
      llvm::Constant *filter_clause = lpi.getClause(current_clause_id);

      if (filter_clause->isNullValue()) {
        // special handling for a catch-all filter clause, i.e., "[0 x i8*]"
        // for this case we serialize 1 element..
        serialized.push_back(1);
        // which is a 64bit-wide 0.
        serialized.resize(serialized.size() + 8, 0);
      } else {
        const auto *ca = cast<llvm::ConstantArray>(filter_clause);

        // serialize `num_elements+1` as unsigned char
        unsigned const num_elements = ca->getNumOperands();
        unsigned char serialized_num_elements = 0;

        if (num_elements >=
            std::numeric_limits<decltype(serialized_num_elements)>::max()) {
          terminateStateOnExecError(
              state, "Internal: too many elements in landingpad filter");
          stateTerminated = true;
          return nullptr;
        }

        serialized_num_elements = num_elements;
        serialized.push_back(serialized_num_elements + 1);

        // serialize the exception-types occurring in this filter-clause
        for (llvm::Value const *v : ca->operands()) {
          llvm::GlobalValue const *clause_value = nullptr;

          if (auto const *bitcast = dyn_cast<llvm::BitCastOperator>(v)) {
            clause_value = dyn_cast<GlobalValue>(bitcast->getOperand(0));
          }

          if (auto const *gv = dyn_cast<llvm::GlobalVariable>(v)) {
            clause_value = gv;
          }

          if (!clause_value) {
            terminateStateOnExecError(state,
                                      "Internal: expected value inside a "
                                      "filter-clause bitcast to be a GlobalValue");
            stateTerminated = true;
            return nullptr;
          }

          std::uint64_t const ti_addr =
              globalAddresses[clause_value]->getZExtValue();

          const std::size_t old_size = serialized.size();
          serialized.resize(old_size + 8);
          memcpy(serialized.data() + old_size, &ti_addr, sizeof(ti_addr));
        }
      }
    }
  }

  MemoryObject *mo =
      memory->allocate(serialized.size(), true, false, &state, nullptr, 1);
  ObjectState *os = bindObjectInState(state, mo, false);
  for (unsigned i = 0; i < serialized.size(); i++) {
    os->write8(i, serialized[i]);
  }

  return mo;
}

void Executor::unwindToNextLandingpad(ExecutionState &state) {
  UnwindingInformation *ui = state.unwindingInformation.get();
  assert(ui && "unwinding without unwinding information");

  std::size_t startIndex;
  std::size_t lowestStackIndex;
  bool popFrames;

  if (auto *sui = dyn_cast<SearchPhaseUnwindingInformation>(ui)) {
    startIndex = sui->unwindingProgress;
    lowestStackIndex = 0;
    popFrames = false;
  } else if (auto *cui = dyn_cast<CleanupPhaseUnwindingInformation>(ui)) {
    startIndex = state.stack.size() - 1;
    lowestStackIndex = cui->catchingStackIndex;
    popFrames = true;
  } else {
    assert(false && "invalid UnwindingInformation subclass");
  }

  for (std::size_t i = startIndex; i > lowestStackIndex; i--) {
    auto const &sf = state.stack.at(i);

    Instruction *inst = sf.caller ? sf.caller->inst : nullptr;

    if (popFrames) {
      state.popFrame();
      if (statsTracker != nullptr) {
        statsTracker->framePopped(state);
      }
    }

    if (InvokeInst *invoke = dyn_cast<InvokeInst>(inst)) {
      // we found the next invoke instruction in the call stack, handle it
      // depending on the current phase.
      if (auto *sui = dyn_cast<SearchPhaseUnwindingInformation>(ui)) {
        // in the search phase, run personality function to check if this
        // landingpad catches the exception

        LandingPadInst *lpi = invoke->getUnwindDest()->getLandingPadInst();
        assert(lpi && "unwind target of an invoke instruction did not lead to "
                      "a landingpad");

        // check if this is a pure cleanup landingpad first
        if (lpi->isCleanup() && lpi->getNumClauses() == 0) {
          // pure cleanup lpi, this can't be a handler, so skip it
          continue;
        }

        bool stateTerminated = false;
        MemoryObject *clauses_mo =
            serializeLandingpad(state, *lpi, stateTerminated);
        assert((stateTerminated != bool(clauses_mo)) &&
               "illegal serializeLandingpad result");

        if (stateTerminated) {
          return;
        }

        assert(sui->serializedLandingpad == nullptr &&
               "serializedLandingpad should be reset");
        sui->serializedLandingpad = clauses_mo;

        llvm::Function *personality_fn =
            kmodule->module->getFunction("_klee_eh_cxx_personality");
        KFunction *kf = kmodule->functionMap[personality_fn];

        state.pushFrame(state.prevPC, kf);
        state.pc = kf->instructions;
        bindArgument(kf, 0, state, sui->exceptionObject);
        bindArgument(kf, 1, state, clauses_mo->getSizeExpr());
        bindArgument(kf, 2, state, clauses_mo->getBaseExpr());

        if (statsTracker) {
          statsTracker->framePushed(state,
                                    &state.stack[state.stack.size() - 2]);
        }

        // make sure we remember our search progress afterwards
        sui->unwindingProgress = i - 1;
      } else {
        // in the cleanup phase, redirect control flow
        transferToBasicBlock(invoke->getUnwindDest(), invoke->getParent(),
                             state);
      }

      // we are done, stop search/unwinding here
      return;
    }
  }

  // no further invoke instruction/landingpad found
  if (isa<SearchPhaseUnwindingInformation>(ui)) {
    // in phase 1, simply stop unwinding. this will return
    // control flow back to _Unwind_RaiseException, which will
    // return the correct error.

    // clean up unwinding state
    state.unwindingInformation.reset();
  } else {
    // in phase 2, this represent a situation that should
    // not happen, as we only progressed to phase 2 because
    // we found a handler in phase 1.
    // therefore terminate the state.
    terminateStateOnExecError(state,
                              "Missing landingpad in phase 2 of unwinding");
  }
}

ref<klee::ConstantExpr> Executor::getEhTypeidFor(ref<Expr> type_info) {
  // FIXME: Handling getEhTypeidFor is non-deterministic and depends on the
  //        order states have been processed and executed.
  auto eh_type_iterator =
      std::find(std::begin(eh_typeids), std::end(eh_typeids), type_info);
  if (eh_type_iterator == std::end(eh_typeids)) {
    eh_typeids.push_back(type_info);
    eh_type_iterator = std::prev(std::end(eh_typeids));
  }
  // +1 because typeids must always be positive, so they can be distinguished
  // from 'no landingpad clause matched' which has value 0
  auto res = ConstantExpr::create(eh_type_iterator - std::begin(eh_typeids) + 1,
                                  Expr::Int32);
  return res;
}

std::string Executor::getSymName(ExecutionState &state, const MemoryObject *mo, ref<Expr> address) {
  if (auto addressCE = dyn_cast<ConstantExpr>(address)) {
    uint64_t destAddress = addressCE->getZExtValue();
    if (destAddress != mo->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        return state.symNames[destAddress];
      }
    }
  }
  return mo->name;
}

void Executor::executeCall(ExecutionState &state, KInstruction *ki, Function *f,
                           std::vector<ref<Expr>> &arguments) {
  if (f->getName().find("c1013intrusive_ptrINS_10TensorImplENS_19UndefinedTensorImpl")!=std::string::npos || 
      f->getName().find("at10TensorBase19unsafeGetTensorImplEv")!=std::string::npos) {
    bindLocal(ki, state, arguments[0]);
    return;
  }

  if (f->getName().find("TensorBase8data_ptr")!=std::string::npos || f->getName().find("TensorBase14const_data_ptr")!=std::string::npos || f->getName().find("TensorBase16mutable_data_ptr")!=std::string::npos || f->getName().find("Tensor8data_ptr")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
      return;
    }
    const MemoryObject *mo = op.first;
    std::string symName = getSymName(state, mo, arguments[0]);
    
    if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
      llvm::Type *returnType = f->getReturnType();

      if (auto sizeCE = dyn_cast<ConstantExpr>(symmo->size)) {
        int size = sizeCE->getZExtValue();
        if (size == 0) {
          bindLocal(ki, state, ConstantExpr::create(0, Expr::Int64));
          return;
        }
      }

      if (symmo->elementSize.isNull()) {
        llvm::Type *pointeeType = returnType->getPointerElementType();
        uint64_t elementSize = getTypeBits(pointeeType) / 8; // bytes
        if (!elementSize) {
          if (auto *structType = llvm::dyn_cast<llvm::StructType>(pointeeType)) {
            if (structType->isSized()) {
              elementSize = kmodule->targetData->getTypeAllocSize(structType);
            }
          }
        }
        if (elementSize) {
          ref<ConstantExpr> elementSizeExpr = ConstantExpr::create(elementSize, Expr::Int64, false);
          symmo->elementSize = elementSizeExpr;
        }
      } 
    }
    bindLocal(ki, state, arguments[0]);
    return;
  }

  if (f->getName().find("detail14torchCheckFail")!=std::string::npos || f->getName().find("c106detail17torchCheckMsgImpl")!=std::string::npos) {
    return;
  }

  if (f->getName().find("detail23torchInternalAssertFail") !=std::string::npos) {
    return;
  }

  if (f->getName().find("__assert_fail")!=std::string::npos) {
    terminateStateOnExecError(state, "reached \"unreachable\" instruction");
    return;
  }

  if (f->getName().find("torch7Library")!=std::string::npos && f->getName().find("Kind")!=std::string::npos && f->getName().find("DispatchKey")!=std::string::npos) {
    return;
  }

  if (f->getName().find("torch6detail16TorchLibraryInit")!=std::string::npos && f->getName().find("7Library4Kind")!=std::string::npos && f->getName().find("DispatchKey")!=std::string::npos) {
    return;
  }

  if (f->getName().find("c104cuda17OptionalCUDAGuard")!=std::string::npos || f->getName().find("ExcludeDispatchKeyGuard")!=std::string::npos) {
    return;
  }

  if (f->getName().find("c1011DeviceGuardC2ENS_6Device")!=std::string::npos) {
    return;
  }

  if (f->getName().find("_ZNSt7__cxx119to_stringE")!=std::string::npos) {
    return;
  }

  if (f->getName().find("St7__cxx1118basic_stringstreamIcSt11char_traitsIcESaIcEE3strEv")!=std::string::npos) {
    return;
  }

  if (f->getName().find("IcSt11char_traitsIcESaIcEE")!=std::string::npos && f->getName().find("NSt7__cxx1112basic_string")!=std::string::npos && f->getReturnType()->isVoidTy()) {
    bool isConstant = true;
    for (auto &arg : arguments) {
      if (!isa<ConstantExpr>(arg)) {
        isConstant = false;
        break;
      }
    }
    if (!isConstant) 
      return;
  }

  if (f->getName()=="_ZN3c103strIJcPKcA24_cS2_A2_cEEEDcDpRKT_") {
    return;
  }

  if (f->getName()=="_ZN5timer3minEv" || f->getName()=="_ZN5timer3maxEv" || f->getName()=="_ZN5timer3meanEv") {
    bindLocal(ki, state, ConstantExpr::create(0, Expr::Int64));
    return;
  }

  if (f->getName().find("at7Context21alertNotDeterministicERKN3c1017basic_string_view")!=std::string::npos) {
    return;
  }

  if (f->getName().find("report_overflow")!=std::string::npos) {
    return;
  }

  if (f->getName().find("9__fill_a1IPllEN9__gnu_cxx11__enable_ifIXsr11__is_scalarIT0_EE7__valueEvE6__typeET_S6_RKS3_")!=std::string::npos || f->getName().find("_ZSt8__fill_aIPllEvT_S1_RKT0_")!=std::string::npos) {
    if (!isa<ConstantExpr>(arguments[0]) || !isa<ConstantExpr>(arguments[1])) return;
  }

  if (f->getName().find("memcmp")!=std::string::npos && arguments.size() == 3) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "do not find object", StateTerminationType::ReportError);
      return;
    }
    const MemoryObject *mo0 = op0.first;
    std::string symName0 = getSymName(state, mo0, arguments[0]);

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "do not find object", StateTerminationType::ReportError);
      return;
    }
    const MemoryObject *mo1 = op1.first;
    std::string symName1 = getSymName(state, mo1, arguments[1]);

    if (!isa<ConstantExpr>(arguments[2])) {
      terminateStateOnProgramError(state, "memcmp length not constant", StateTerminationType::ReportError);
      return;
    }

    auto lengthCE = dyn_cast<ConstantExpr>(arguments[2]);
    int length = lengthCE->getZExtValue();

    if (state.tensorSizesMap.find(mo0->name)!=state.tensorSizesMap.end() || state.tensorSizesMap.find(mo1->name)!=state.tensorSizesMap.end()) {
      std::vector<ref<Expr>> sizes0;
      if (state.tensorSizesMap.find(mo0->name)!=state.tensorSizesMap.end()) {
        TensorSizesMemoryObject *tensorSizesMO0 = state.tensorSizesMap[mo0->name];
        if (state.symbolicArrayMap.find(tensorSizesMO0->tensorName)==state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "did not find tensor object", StateTerminationType::ReportError);
          return;
        }
        if (auto shapeSizeCE = dyn_cast<ConstantExpr>(tensorSizesMO0->length)) {
          int shapeSize = shapeSizeCE->getZExtValue();
          for (int i = 0; i < shapeSize; i++) {
            sizes0.push_back(tensorSizesMO0->elements[i]);
          }
        }
      } else {
        const ObjectState *os0 = op0.second;
        for (int i = 0; i < length; i++) {
          ref<Expr> val = os0->read(i*8, Expr::Int64);
          sizes0.push_back(val);
        }
      }

      std::vector<ref<Expr>> sizes1;
      if (state.tensorSizesMap.find(mo1->name)!=state.tensorSizesMap.end()) {
        TensorSizesMemoryObject *tensorSizesMO1 = state.tensorSizesMap[mo1->name];
        if (state.symbolicArrayMap.find(tensorSizesMO1->tensorName)==state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "did not find tensor object", StateTerminationType::ReportError);
          return;
        }
        if (auto shapeSizeCE = dyn_cast<ConstantExpr>(tensorSizesMO1->length)) {
          int shapeSize = shapeSizeCE->getZExtValue();
          for (int i = 0; i < shapeSize; i++) {
            sizes1.push_back(tensorSizesMO1->elements[i]);
          }
        }
      } else {
        const ObjectState *os1 = op1.second;
        for (int i = 0; i < length; i++) {
          ref<Expr> val = os1->read(i*8, Expr::Int64);
          sizes1.push_back(val);
        }
      }

      ref<Expr> result = nullptr;
      for (int i = 0; i < length; i++) {
        if (i==0) {
          result = EqExpr::create(ZExtExpr::create(sizes0[i], Expr::Int64), ZExtExpr::create(sizes1[i], Expr::Int64));
        } else {
          result = AndExpr::create(result, EqExpr::create(ZExtExpr::create(sizes0[i], Expr::Int64), ZExtExpr::create(sizes1[i], Expr::Int64)));
        }
      }
      llvm::Type *returnType = f->getReturnType();
      result = ZExtExpr::create(NotExpr::create(result), returnType->getIntegerBitWidth());
      bindLocal(ki, state, result);
      return;
    }
  }

  if (f->getName()=="memcpy") {
    if (auto bytesCE = dyn_cast<ConstantExpr>(arguments[2])) {
      unsigned bytes = bytesCE->getZExtValue();
      executeMemoryOperation(state, false, arguments[1], 0, ki, bytes);

      ref<Expr> loadedVal = getDestCell(state, ki).value;
      if (!loadedVal.isNull()) {
        executeMemoryOperation(state, true, arguments[0], loadedVal, ki);
        getDestCell(state, ki).value = arguments[0];
        return;
      } else {
        klee_warning("memcpy load value is null");
      }
    } else {
      klee_warning("memcpy bytes are not constant");
    }
  }

  if (f->getName().find("4cuda3__414__memcpy_async") != std::string::npos) {
    if (auto bytesCE = dyn_cast<ConstantExpr>(arguments[3])) {
      unsigned bytes = bytesCE->getZExtValue();
      executeMemoryOperation(state, false, arguments[2], 0, ki, bytes);

      ref<Expr> loadedVal = getDestCell(state, ki).value;
      if (!loadedVal.isNull()) {
        executeMemoryOperation(state, true, arguments[1], loadedVal, ki);
        getDestCell(state, ki).value = ConstantExpr::create(0, Expr::Int32);
        return;
      } else {
        klee_warning("cuda memcpy_async load value is null");
      }
    } else {
      klee_warning("cuda memcpy_async bytes are not constant");
      executeCudaMemcpy(state, arguments[1], arguments[2], arguments[3], ki);
      bindLocal(ki, state, ConstantExpr::create(0, Expr::Int32));
      return;
    }
  }
                            
  Instruction *i = ki->inst;
  if (isa_and_nonnull<DbgInfoIntrinsic>(i))
    return;
  if (f && f->isDeclaration()) {
    switch (f->getIntrinsicID()) {
    case Intrinsic::not_intrinsic: {
      // state may be destroyed by this call, cannot touch
      callExternalFunction(state, ki, kmodule->functionMap[f], arguments);
      break;
    }
    case Intrinsic::fabs: {
      ref<ConstantExpr> arg =
          toConstant(state, arguments[0], "floating point");
      if (!fpWidthToSemantics(arg->getWidth()))
        return terminateStateOnExecError(
            state, "Unsupported intrinsic llvm.fabs call");

      llvm::APFloat Res(*fpWidthToSemantics(arg->getWidth()),
                        arg->getAPValue());
      Res = llvm::abs(Res);

      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), false, true));
      break;
    }

    case Intrinsic::fma:
    case Intrinsic::fmuladd: {
      // Both fma and fmuladd support float, double and fp80.  Note, that fp80
      // is not mentioned in the documentation of fmuladd, nevertheless, it is
      // still supported.  For details see
      // https://github.com/klee/klee/pull/1507/files#r894993332

      if (isa<VectorType>(i->getOperand(0)->getType()))
        return terminateStateOnExecError(
            state, f->getName() + " with vectors is not supported");

      ref<ConstantExpr> op1 =
          toConstant(state, eval(ki, 1, state).value, "floating point");
      ref<ConstantExpr> op2 =
          toConstant(state, eval(ki, 2, state).value, "floating point");
      ref<ConstantExpr> op3 =
          toConstant(state, eval(ki, 3, state).value, "floating point");

      if (!fpWidthToSemantics(op1->getWidth()) ||
          !fpWidthToSemantics(op2->getWidth()) ||
          !fpWidthToSemantics(op3->getWidth()))
        return terminateStateOnExecError(
            state, "Unsupported " + f->getName() + " call");

      APFloat Res(*fpWidthToSemantics(op1->getWidth()), op1->getAPValue());
      Res.fusedMultiplyAdd(
          APFloat(*fpWidthToSemantics(op2->getWidth()), op2->getAPValue()),
          APFloat(*fpWidthToSemantics(op3->getWidth()), op3->getAPValue()),
          APFloat::rmNearestTiesToEven);

      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
      break;
    }

#if LLVM_VERSION_CODE >= LLVM_VERSION(12, 0)
    case Intrinsic::abs: {
      if (isa<VectorType>(i->getOperand(0)->getType()))
        return terminateStateOnExecError(
            state, "llvm.abs with vectors is not supported");

      ref<Expr> op = eval(ki, 1, state).value;
      ref<Expr> poison = eval(ki, 2, state).value;

      assert(poison->getWidth() == 1 && "Second argument is not an i1");
      unsigned bw = op->getWidth();

      uint64_t moneVal = APInt(bw, -1, true).getZExtValue();
      uint64_t sminVal = APInt::getSignedMinValue(bw).getZExtValue();

      ref<ConstantExpr> zero = ConstantExpr::create(0, bw);
      ref<ConstantExpr> mone = ConstantExpr::create(moneVal, bw);
      ref<ConstantExpr> smin = ConstantExpr::create(sminVal, bw, true);

      if (poison->isTrue()) {
        ref<Expr> issmin = EqExpr::create(op, smin);
        if (issmin->isTrue())
          return terminateStateOnExecError(
              state, "llvm.abs called with poison and INT_MIN");
      }

      ref<Expr> negative = SltExpr::create(op, zero);
      ref<Expr> notsmin = NeExpr::create(op, smin);
      ref<Expr> cond = AndExpr::create(negative, notsmin);

      // flip and select the result
      ref<Expr> flip = MulExpr::create(op, mone);
      ref<Expr> result = SelectExpr::create(cond, flip, op);

      bindLocal(ki, state, result);
      break;
    }

    case Intrinsic::smax:
    case Intrinsic::smin:
    case Intrinsic::umax:
    case Intrinsic::umin: {
      if (isa<VectorType>(i->getOperand(0)->getType()) ||
          isa<VectorType>(i->getOperand(1)->getType()))
        return terminateStateOnExecError(
            state, "llvm.{s,u}{max,min} with vectors is not supported");

      ref<Expr> op1 = eval(ki, 1, state).value;
      ref<Expr> op2 = eval(ki, 2, state).value;

      ref<Expr> cond = nullptr;
      if (f->getIntrinsicID() == Intrinsic::smax)
        cond = SgtExpr::create(op1, op2);
      else if (f->getIntrinsicID() == Intrinsic::smin)
        cond = SltExpr::create(op1, op2);
      else if (f->getIntrinsicID() == Intrinsic::umax)
        cond = UgtExpr::create(op1, op2);
      else // (f->getIntrinsicID() == Intrinsic::umin)
        cond = UltExpr::create(op1, op2);

      ref<Expr> result = SelectExpr::create(cond, op1, op2);
      bindLocal(ki, state, result);
      break;
    }
#endif

    case Intrinsic::fshr:
    case Intrinsic::fshl: {
      ref<Expr> op1 = eval(ki, 1, state).value;
      ref<Expr> op2 = eval(ki, 2, state).value;
      ref<Expr> op3 = eval(ki, 3, state).value;
      unsigned w = op1->getWidth();
      assert(w == op2->getWidth() && "type mismatch");
      assert(w == op3->getWidth() && "type mismatch");
      ref<Expr> c = ConcatExpr::create(op1, op2);
      op3 = URemExpr::create(op3, ConstantExpr::create(w, w));
      op3 = ZExtExpr::create(op3, w+w);
      if (f->getIntrinsicID() == Intrinsic::fshl) {
        // shift left and take top half
        ref<Expr> s = ShlExpr::create(c, op3);
        bindLocal(ki, state, ExtractExpr::create(s, w, w));
      } else {
        // shift right and take bottom half
        // note that LShr and AShr will have same behaviour
        ref<Expr> s = LShrExpr::create(c, op3);
        bindLocal(ki, state, ExtractExpr::create(s, 0, w));
      }
      break;
    }

    // va_arg is handled by caller and intrinsic lowering, see comment for
    case Intrinsic::vastart: {
      StackFrame &sf = state.stack.back();

      // varargs can be zero if no varargs were provided
      if (!sf.varargs)
        return;

      // FIXME: This is really specific to the architecture, not the pointer
      // size. This happens to work for x86-32 and x86-64, however.
      Expr::Width WordSize = Context::get().getPointerWidth();
      if (WordSize == Expr::Int32) {
        executeMemoryOperation(state, true, arguments[0],
                               sf.varargs->getBaseExpr(), 0);
      } else {
        assert(WordSize == Expr::Int64 && "Unknown word size!");

        // x86-64 has quite complicated calling convention. However,
        // instead of implementing it, we can do a simple hack: just
        // make a function believe that all varargs are on stack.
        executeMemoryOperation(
            state, true, 
            arguments[0],
            ConstantExpr::create(48, 32), 0); // gp_offset
        executeMemoryOperation(
            state, true,
            AddExpr::create(arguments[0], ConstantExpr::create(4, 64)),
            ConstantExpr::create(304, 32), 0); // fp_offset
        executeMemoryOperation(
            state, true,
            AddExpr::create(arguments[0], ConstantExpr::create(8, 64)),
            sf.varargs->getBaseExpr(), 0); // overflow_arg_area
        executeMemoryOperation(
            state, true,
            AddExpr::create(arguments[0], ConstantExpr::create(16, 64)),
            ConstantExpr::create(0, 64), 0); // reg_save_area
      }
      break;
    }

#ifdef SUPPORT_KLEE_EH_CXX
    case Intrinsic::eh_typeid_for: {
      bindLocal(ki, state, getEhTypeidFor(arguments.at(0)));
      break;
    }
#endif

    case Intrinsic::vaend:
      // va_end is a noop for the interpreter.
      //
      // FIXME: We should validate that the target didn't do something bad
      break;

    case Intrinsic::vacopy:
      // va_copy should have been lowered.
      //
      // FIXME: It would be nice to check for errors in the usage of this as
      // well.
    
    
    default:
      if (f->getName() == "llvm.nvvm.read.ptx.sreg.ctaid.x") {
        ref<Expr> indexExpr = state.kernelConfig.getBlockIdx().x;
        bindLocal(ki, state, indexExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.ctaid.y") {
        ref<Expr> indexExpr = state.kernelConfig.getBlockIdx().y;
        bindLocal(ki, state, indexExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.ctaid.z") {
        ref<Expr> indexExpr = state.kernelConfig.getBlockIdx().z;
        bindLocal(ki, state, indexExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
        ref<Expr> indexExpr = state.kernelConfig.getThreadIdx().x;
        bindLocal(ki, state, indexExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.tid.y") {
        ref<Expr> indexExpr = state.kernelConfig.getThreadIdx().y;
        bindLocal(ki, state, indexExpr);
      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.tid.z") {
        ref<Expr> indexExpr = state.kernelConfig.getThreadIdx().z;
        bindLocal(ki, state, indexExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.nctaid.x") {
        ref<Expr> dimExpr = state.kernelConfig.getGridDim().x;
        bindLocal(ki, state, dimExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.nctaid.y") {
        ref<Expr> dimExpr = state.kernelConfig.getGridDim().y;
        bindLocal(ki, state, dimExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.nctaid.z") {
        ref<Expr> dimExpr = state.kernelConfig.getGridDim().z;
        bindLocal(ki, state, dimExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.ntid.x") {
        ref<Expr> dimExpr = state.kernelConfig.getBlockDim().x;
        bindLocal(ki, state, dimExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.ntid.y") {
        ref<Expr> dimExpr = state.kernelConfig.getBlockDim().y;
        bindLocal(ki, state, dimExpr);

      } else if (f->getName() == "llvm.nvvm.read.ptx.sreg.ntid.z") {
        ref<Expr> dimExpr = state.kernelConfig.getBlockDim().z;
        bindLocal(ki, state, dimExpr);
        
      } else if (f->getName().find("llvm.nvvm.ldg.global")==0) {
        executeMemoryOperation(state, false, arguments[0], 0, ki);
      } else if (f->getName() == "llvm.nvvm.barrier0" || f->getName() == "llvm.nvvm.bar.warp.sync" || f->getName() == "llvm.nvvm.barrier.sync") {
      } else if (f->getName().find("llvm.nvvm.shfl.sync")==0) {
        bindLocal(ki, state, arguments[1]);
      } else if (f->getName() == "llvm.nvvm.rsqrt.approx.f" || f->getName() == "llvm.nvvm.sqrt.approx.f") {
        bindLocal(ki, state, arguments[0]); // should be rsqrt(arguments[0]), but do not care about the precision
      } else if (f->getName() == "llvm.nvvm.fma.rn.f" || f->getName() == "llvm.nvvm.fma.rm.f" || f->getName() == "llvm.nvvm.fma.rn.d") {
        bindLocal(ki, state, AddExpr::create(MulExpr::create(arguments[0], arguments[1]), arguments[2]));
      } else if (f->getName() == "llvm.nvvm.saturate.f") {
        bindLocal(ki, state, ConstantExpr::create(1, f->getReturnType()->getPrimitiveSizeInBits()));
      } else if (f->getName().find("llvm.nvvm.ex2.approx")==0) {
        bindLocal(ki, state, ConstantExpr::create(4, f->getReturnType()->getPrimitiveSizeInBits()));
      } else if (f->getName() == "llvm.nvvm.fabs.f") {
        bindLocal(ki, state, ConstantExpr::create(2, f->getReturnType()->getPrimitiveSizeInBits()));
      } else if (f->getName()=="llvm.nvvm.fmax.f" || f->getName()=="llvm.nvvm.fmin.f" || f->getName()=="llvm.nvvm.fmax.d" || f->getName()=="llvm.nvvm.fmin.d") {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.add.rz.f" || f->getName()=="llvm.nvvm.add.rn.f") {
        bindLocal(ki, state, AddExpr::create(arguments[0], arguments[1]));
      } else if (f->getName()=="llvm.nvvm.trunc.f" || f->getName().find("llvm.nearbyint")==0) {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.mul.rn.f") {
        bindLocal(ki, state, MulExpr::create(arguments[0], arguments[1]));
      } else if (f->getName()=="llvm.nvvm.div.approx.f") { // @TODO, round up
        bindLocal(ki, state, SDivExpr::create(arguments[0], arguments[1]));
      } else if (f->getName()=="llvm.nvvm.prmt") {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.f2i.rn") {
        if (auto CE = dyn_cast<ConstantExpr>(arguments[0])) {
          uint32_t floatBits = CE->getZExtValue();
          float float_val;
          std::memcpy(&float_val, &floatBits, sizeof(float_val));
          int32_t result = static_cast<int32_t>(std::nearbyint(float_val));
          bindLocal(ki, state, ConstantExpr::create(result, Expr::Int32));
        } else
          bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.sqrt.rn.d") { // TODO
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.floor.f" || f->getName()=="llvm.nvvm.ceil.f") {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.read.ptx.sreg.clock64") {
        bindLocal(ki, state, ConstantExpr::create(1, Expr::Int64));
      } else if (f->getName()=="llvm.nvvm.mulhi.ui") {
        ref<Expr> a64 = ZExtExpr::create(arguments[0], Expr::Int64);
        ref<Expr> b64 = ZExtExpr::create(arguments[1], Expr::Int64);
        // Multiply: 64-bit result
        ref<Expr> prod64 = MulExpr::create(a64, b64);
        // Shift right by 32 to get the high 32 bits
        ref<Expr> high32 = LShrExpr::create(prod64, ConstantExpr::alloc(32, Expr::Int64));
        bindLocal(ki, state, ExtractExpr::create(high32, 0, Expr::Int32));
      } else if (f->getName()=="llvm.nvvm.fabs.d") {
        bindLocal(ki, state, ConstantExpr::create(1, Expr::Int64));
      } else if (f->getName()=="llvm.nvvm.ui2f.rn") {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.d2ll.rm") {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.i2f.rz") {
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.d2i.hi") {
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(arguments[0])) {
          uint64_t bits = CE->getZExtValue(); 
          double val;
          memcpy(&val, &bits, sizeof(double));
          // Convert double to int64_t rounding to nearest integer
          int64_t intVal = static_cast<int64_t>(std::llround(val));
          uint32_t high32 = (uint32_t)((intVal >> 32) & 0xFFFFFFFF);
          bindLocal(ki, state, ConstantExpr::create(high32, Expr::Int32));
        } else {
          bindLocal(ki, state, ExtractExpr::create(arguments[0], 0, Expr::Int32));
        }
      } else if (f->getName()=="llvm.nvvm.d2i.lo") {
        double concreteVal;
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(arguments[0])) {
          uint64_t bits = CE->getZExtValue(); 
          double val;
          memcpy(&val, &bits, sizeof(double));
          concreteVal = val;
          // Round to nearest 64-bit int
          int64_t intVal = (int64_t)std::llround(concreteVal);
          // Extract low 32 bits
          uint32_t low32 = (uint32_t)(intVal & 0xFFFFFFFF);
          bindLocal(ki, state, ConstantExpr::create(low32, Expr::Int32));
        } else {
          bindLocal(ki, state, ExtractExpr::create(arguments[0], 32, Expr::Int32));
        }
      } else if (f->getName()=="llvm.nvvm.i2f.rp" || f->getName()=="llvm.nvvm.i2f.rm") { 
        bindLocal(ki, state, arguments[0]);
      } else if (f->getName()=="llvm.nvvm.vote.ballot.sync") { 
        bindLocal(ki, state, ConstantExpr::create(1, Expr::Int32));
      } else {
        klee_warning("unimplemented intrinsic: %s", f->getName().data());
        terminateStateOnExecError(state, "unimplemented intrinsic");
        return;
      }
    }

    if (InvokeInst *ii = dyn_cast<InvokeInst>(i)) {
      transferToBasicBlock(ii->getNormalDest(), i->getParent(), state);
    }
  } else {
    if (f->getName().find("cudaError9cudaErrorPKci") != std::string::npos && f->getReturnType()->isVoidTy()) {
      return;
    } else if (f->getName().find("c106detail25fp8e4m3fn_from_fp32_valueEf") != std::string::npos) {
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("vllm_is_batch_invariant")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorBase5numel")!=std::string::npos || f->getName().find("Tensor5numel")!=std::string::npos || f->getName().find("c1010TensorImpl5numel")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        llvm::Type *returnType = f->getReturnType();
        bindLocal(ki, state, ZExtExpr::create(state.symbolicArrayMap[symName]->size, returnType->getIntegerBitWidth()));
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at10TensorBase9sym_numelEv")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "do not find return object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo1 = op1.first;
      std::string symName = getSymName(state, mo1, arguments[1]);
      ObjectState *os0 = state.addressSpace.getWriteable(op0.first, op0.second);
      
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        os0->write(0, ZExtExpr::create(state.symbolicArrayMap[symName]->size, Expr::Int64));
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("c1010TensorImpl26has_symbolic_sizes_stridesEv")!=std::string::npos) {
      bindLocal(ki, state, ConstantExpr::create(1, Expr::Bool));
      return;
    } else if (f->getName().find("TensorIteratorBase8ntensors")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      if (state.inputIterator->size.isNull()) {
        if (!state.inputIterator->tensors.size()) {
          terminateStateOnProgramError(state, "TensorIterator.size()==0", StateTerminationType::ReportError);
          return;
        }
        state.inputIterator->size = ConstantExpr::create(state.inputIterator->tensors.size(), returnType->getIntegerBitWidth());
      }
      bindLocal(ki, state, ZExtExpr::create(state.inputIterator->size, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorIteratorBase8noutputs")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(state.inputIterator->noutputs, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorIteratorBase4ndimEv")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      computeTensorIteratorShape(state);
      bindLocal(ki, state, ConstantExpr::create(state.inputIterator->shape.size(), returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorBase4size")!=std::string::npos || f->getName().find("Tensor4size")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        ref<Expr> indexExpr = arguments[1];
        if (auto ce = dyn_cast<ConstantExpr>(indexExpr)) {
          llvm::Type *returnType = f->getReturnType();
          unsigned width = returnType->getIntegerBitWidth();
          int index = ce->getZExtValue();
          SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];

          if (index < 0 && !symmo->shapeSize.isNull() && isa<ConstantExpr>(symmo->shapeSize)) {
            auto shapeSizeCE = dyn_cast<ConstantExpr>(symmo->shapeSize);
            int shapeSize = shapeSizeCE->getZExtValue();
            index = shapeSize + index;
          }

          if (symmo->dimensionSize.find(index) == symmo->dimensionSize.end() || symmo->dimensionSize[index].isNull()) {
            std::string sizeName = symName + ".size[" + std::to_string(index)+"]";
            auto sizepair = createSizeSymbol(state, sizeName);
            ref<Expr> sizeExpr = sizepair.second; 
            addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));  
            symmo->dimensionSize[index] = sizeExpr;
          }
          bindLocal(ki, state, ZExtExpr::create(symmo->dimensionSize[index], width));
          return;
        } else {
          terminateStateOnProgramError(state, "TensorBase4size dimension index not constant", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at10TensorBase3dim")!=std::string::npos || f->getName().find("Tensor3dim")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        llvm::Type *returnType = f->getReturnType();
        unsigned width = returnType->getIntegerBitWidth();
        if (symmo->shapeSize.isNull()) {
          std::string dimName = symName + ".dim";
          auto dimpair = createSizeSymbol(state, dimName);
          ref<Expr> dimExpr = dimpair.second; 
          symmo->shapeSize = dimExpr;
        }
        bindLocal(ki, state, ZExtExpr::create(symmo->shapeSize, width));
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("TensorBase5sizes")!=std::string::npos || f->getName().find("at10TensorBase9sym_sizesEv")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        std::string sizesName = symName + ".sizes()";

        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if (symmo->shapeSize.isNull()) {
          std::string dimName = symName + ".dim";
          auto dimpair = createSizeSymbol(state, dimName);
          ref<Expr> dimExpr = dimpair.second; 
          symmo->shapeSize = dimExpr;
        }

        uint64_t arrayAddress;
        if (state.tensorSizesMap.find(sizesName) != state.tensorSizesMap.end()) {
          arrayAddress = state.tensorSizesMap[sizesName]->address;
        } else {
          MemoryObject *sizesMo =
            memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
          sizesMo->setName(sizesName);
          const Array *sizesArray = arrayCache.CreateArray(sizesName, sizesMo->size);
          bindObjectInState(state, sizesMo, false, sizesArray);
          state.addSymbolic(sizesMo, sizesArray);
          TensorSizesMemoryObject* tensorSizesSymMO = new TensorSizesMemoryObject(sizesMo->address, sizesArray, symName, symmo->shapeSize);
          state.tensorSizesMap[sizesName] = tensorSizesSymMO;
          tensorSizesSymMO->elements = symmo->dimensionSize;
          arrayAddress = sizesMo->address;
        }

        ref<Expr> structExpr = ConcatExpr::create(ZExtExpr::create(symmo->shapeSize, Expr::Int64), ConstantExpr::create(arrayAddress, 64));
        bindLocal(ki, state, structExpr);
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("TensorBase7strides")!=std::string::npos || f->getName().find("at10TensorBase11sym_stridesEv")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        std::string stridesName = symName + ".strides()";

        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if (symmo->shapeSize.isNull()) {
          std::string dimName = symName + ".dim";
          auto dimpair = createSizeSymbol(state, dimName);
          ref<Expr> dimExpr = dimpair.second; 
          symmo->shapeSize = dimExpr;
        }

        uint64_t arrayAddress;
        if (state.tensorSizesMap.find(stridesName) != state.tensorSizesMap.end()) {
          arrayAddress = state.tensorSizesMap[stridesName]->address;
        } else {
          MemoryObject *stridesMo =
            memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
          stridesMo->setName(stridesName);
          const Array *stridesArray = arrayCache.CreateArray(stridesName, stridesMo->size);
          bindObjectInState(state, stridesMo, false, stridesArray);
          state.addSymbolic(stridesMo, stridesArray);
          TensorSizesMemoryObject* tensorSizesSymMO = new TensorSizesMemoryObject(stridesMo->address, stridesArray, symName, symmo->shapeSize);
          state.tensorSizesMap[stridesName] = tensorSizesSymMO;
          tensorSizesSymMO->elements = symmo->strides;
          arrayAddress = stridesMo->address;
        }

        ref<Expr> structExpr = ConcatExpr::create(ZExtExpr::create(symmo->shapeSize, Expr::Int64), ConstantExpr::create(arrayAddress, 64));
        bindLocal(ki, state, structExpr);
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("TensorBase6stride")!=std::string::npos || f->getName().find("Tensor6stride")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        ref<Expr> indexExpr = arguments[1];
        indexExpr = ConstraintManager::simplifyExpr(state.constraints, indexExpr);
        if (auto ce = dyn_cast<ConstantExpr>(indexExpr)) {
          llvm::Type *returnType = f->getReturnType();
          unsigned width = returnType->getIntegerBitWidth();
          int index = ce->getZExtValue();

          SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
          if (symmo->strides.find(index) != symmo->strides.end()) {
            bindLocal(ki, state, ZExtExpr::create(symmo->strides[index], width));
            return;
          }
          ref<Expr> strideRes = nullptr;
          if (!symmo->shapeSize.isNull() && isa<ConstantExpr>(symmo->shapeSize)) {
            auto shapeSizeExpr = dyn_cast<ConstantExpr>(symmo->shapeSize);
            uint64_t shapeSize = shapeSizeExpr->getZExtValue();
            if (index < 0) {
              index = shapeSize + index;
            }
            strideRes = ConstantExpr::create(1, width);
            for (uint64_t i = index+1; i < shapeSize; i++) {
              ref<Expr> dimSize; 
              if (symmo->dimensionSize.find(i) != symmo->dimensionSize.end()) {
                dimSize = symmo->dimensionSize[i];
              } else {
                std::string sizeName = symName + ".size[" + std::to_string(i)+"]";
                auto sizepair = createSizeSymbol(state, sizeName);
                dimSize = sizepair.second; 
                addConstraint(state, UgeExpr::create(dimSize, ConstantExpr::create(1, dimSize->getWidth())));
                symmo->dimensionSize[i] = dimSize;
              }
              strideRes = MulExpr::create(strideRes, ZExtExpr::create(dimSize, width));
            }
          } else if (index < 0) {
            strideRes = ConstantExpr::create(1, width);
            for (int i = -1; i > index; i--) {
              ref<Expr> dimSize; 
              if (symmo->dimensionSize.find(i) != symmo->dimensionSize.end()) {
                dimSize = symmo->dimensionSize[i];
              } else {
                std::string sizeName = symName + ".size[" + std::to_string(i)+"]";
                auto sizepair = createSizeSymbol(state, sizeName);
                dimSize = sizepair.second; 
                addConstraint(state, UgeExpr::create(dimSize, ConstantExpr::create(1, dimSize->getWidth())));  
                symmo->dimensionSize[i] = dimSize;
              }
              strideRes = MulExpr::create(strideRes, ZExtExpr::create(dimSize, width));
            }
          } else {
            std::string strideName = symName + ".stride[" + std::to_string(index)+"]";
            auto stridepair = createSizeSymbol(state, strideName, true);
            strideRes = ZExtExpr::create(stridepair.second, width); 
          }
          
          if (strideRes.isNull()) {
            terminateStateOnProgramError(state, "tensor stride index "+std::to_string(index)+" not handled", StateTerminationType::ReportError);
            return;
          }
          symmo->strides[index] = strideRes;
          bindLocal(ki, state, strideRes);
          return;
        } else {
          llvm::Type *returnType = f->getReturnType();
          unsigned width = returnType->getIntegerBitWidth();
          std::string strideName = symName + ".stride[" + std::to_string(indexExpr->count)+"]";
          auto stridepair = createSizeSymbol(state, strideName, true);
          ref<Expr> strideRes = ZExtExpr::create(stridepair.second, width); 
          bindLocal(ki, state, strideRes);
          klee_warning("TensorBase6stride stride index not constant for tensor %s", symName.c_str());
          return;
        }
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at10TensorBase12element_size")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        llvm::Type *returnType = f->getReturnType();
        unsigned width = returnType->getIntegerBitWidth();
        setElementSize(state, symmo);

        if (symmo->elementSize && isa<ConstantExpr>(symmo->elementSize)) {
          bindLocal(ki, state, ZExtExpr::create(symmo->elementSize, width));
        } else if (symmo->scalarType && isa<ConstantExpr>(symmo->scalarType)) {
          unsigned size = getTensorTypeBytes(symmo->scalarType);
          if (size == 0) {
            klee_warning("tensor.element_size: not handled type");
            std::string sizeName = symName + ".item_size";
            MemoryObject *sizeMo =
              memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                              &state, /*allocSite=*/state.prevPC->inst,
                              /*alignment=*/8);          
            ref<Expr> sizeExpr = symbolizeIndex(state, sizeName, sizeMo);
            symmo->elementSize = sizeExpr;
            addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
            addConstraint(state, UleExpr::create(sizeExpr, ConstantExpr::create(16, sizeExpr->getWidth())));
            bindLocal(ki, state, ZExtExpr::create(sizeExpr, width));
          } else {
            symmo->elementSize = ConstantExpr::create(size, width);
            bindLocal(ki, state, symmo->elementSize);
          }
        } else if (symmo->elementSize) {
          klee_warning("tensor element size not constant");
          bindLocal(ki, state, ZExtExpr::create(symmo->elementSize, width));
        } else {
          klee_warning("tensor element size not constant");
          std::string sizeName = symName + ".item_size";
          MemoryObject *sizeMo =
            memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
          ref<Expr> sizeExpr = symbolizeIndex(state, sizeName, sizeMo);
          symmo->elementSize = sizeExpr;
          addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
          addConstraint(state, UleExpr::create(sizeExpr, ConstantExpr::create(16, sizeExpr->getWidth())));
          bindLocal(ki, state, ZExtExpr::create(sizeExpr, width));
        }
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("device_of")!=std::string::npos && f->getName().find("Tensor")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorBase10get_device")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("16get_device_index")!=std::string::npos && f->getName().find("Tensor")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorBase7is_cuda")!=std::string::npos || f->getName().find("2at24DeprecatedTypeProperties7is_cuda")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("3c106Device6is_cpuEv")!=std::string::npos) {
      bindLocal(ki, state, ConstantExpr::create(1, Expr::Bool));
      return;
    } else if (f->getName().find("Device8validateEv")!=std::string::npos) {
      return;
    } else if (f->getName().find("at10TensorBase18sym_storage_offsetEv")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find return object", StateTerminationType::ReportError);
        return;
      }
      ObjectState *os = state.addressSpace.getWriteable(op.first, op.second);
      os->write(0, ConstantExpr::create(0, 64));
      return;
    } else if (f->getName().find("at10TensorBase12is_quantized")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        llvm::Type *returnType = f->getReturnType();
        unsigned width = returnType->getIntegerBitWidth();
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if (!symmo->isQuantized) {
          std::string boolName = symName + ".is_quantized";
          MemoryObject *boolMo =
                memory->allocate(1, /*isLocal=*/false, /*isGlobal=*/false,
                                &state, /*allocSite=*/state.prevPC->inst,
                                /*alignment=*/8);          
          boolMo->setName(boolName);
          const Array *boolArray = arrayCache.CreateArray(boolName, boolMo->size, true);
          bindObjectInState(state, boolMo, false, boolArray);
          state.addSymbolic(boolMo, boolArray);
          ref<Expr> quantizedExpr = Expr::createTempRead(boolArray, boolMo->size*8);
          addConstraint(state, UgeExpr::create(quantizedExpr, ConstantExpr::create(0, quantizedExpr->getWidth())));
          addConstraint(state, UleExpr::create(quantizedExpr, ConstantExpr::create(1, quantizedExpr->getWidth())));
          symmo->isQuantized = quantizedExpr;
        }
        bindLocal(ki, state, ZExtExpr::create(symmo->isQuantized, width));
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at6Tensor7qscheme")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        llvm::Type *returnType = f->getReturnType();
        unsigned width = returnType->getIntegerBitWidth();
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if (!symmo->qscheme) {
          std::string schemeName = symName + ".qscheme";
          MemoryObject *schemeMo =
                memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                                &state, /*allocSite=*/state.prevPC->inst,
                                /*alignment=*/8);          
          schemeMo->setName(schemeName);
          const Array *schemeArray = arrayCache.CreateArray(schemeName, schemeMo->size, true);
          bindObjectInState(state, schemeMo, false, schemeArray);
          state.addSymbolic(schemeMo, schemeArray);
          ref<Expr> schemeExpr = Expr::createTempRead(schemeArray, schemeArray->size*8);
          addConstraint(state, UgeExpr::create(schemeExpr, ConstantExpr::create(0, schemeExpr->getWidth())));
          addConstraint(state, UleExpr::create(schemeExpr, ConstantExpr::create(5, schemeExpr->getWidth())));
          symmo->qscheme = schemeExpr;
        }
        bindLocal(ki, state, ZExtExpr::create(symmo->qscheme, width));
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("TensorBase11scalar_type")!=std::string::npos || f->getName().find("TensorBase5dtype")!=std::string::npos || f->getName().find("Tensor11scalar_type")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        llvm::Type *returnType = f->getReturnType();
        unsigned width = returnType->getIntegerBitWidth();
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if (!symmo->scalarType) {
          std::string typeName = symName + ".scalar_type";
          MemoryObject *typeMo =
                memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                                &state, /*allocSite=*/state.prevPC->inst,
                                /*alignment=*/8);          
          typeMo->setName(typeName);
          const Array *typeArray = arrayCache.CreateArray(typeName, typeMo->size, true);
          bindObjectInState(state, typeMo, false, typeArray);
          state.addSymbolic(typeMo, typeArray);
          ref<Expr> scalarTypeExpr = Expr::createTempRead(typeArray, typeMo->size*8);
          addConstraint(state, UgeExpr::create(scalarTypeExpr, ConstantExpr::create(0, scalarTypeExpr->getWidth())));
          addConstraint(state, UleExpr::create(scalarTypeExpr, ConstantExpr::create(255, scalarTypeExpr->getWidth())));
          symmo->scalarType = scalarTypeExpr;
        }
        bindLocal(ki, state, ZExtExpr::create(symmo->scalarType, width));
        return;
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("2at6Tensor4type")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if (!symmo->scalarType) {
          std::string typeName = symName + ".scalar_type";
          MemoryObject *typeMo =
                memory->allocate(1, /*isLocal=*/false, /*isGlobal=*/false,
                                &state, /*allocSite=*/state.prevPC->inst,
                                /*alignment=*/8);          
          typeMo->setName(typeName);
          const Array *typeArray = arrayCache.CreateArray(typeName, typeMo->size, true);
          bindObjectInState(state, typeMo, false, typeArray);
          state.addSymbolic(typeMo, typeArray);
          ref<Expr> scalarTypeExpr = Expr::createTempRead(typeArray, typeMo->size*8);
          addConstraint(state, UgeExpr::create(scalarTypeExpr, ConstantExpr::create(0, scalarTypeExpr->getWidth())));
          addConstraint(state, UleExpr::create(scalarTypeExpr, ConstantExpr::create(255, scalarTypeExpr->getWidth())));
          symmo->scalarType = scalarTypeExpr;
        }
        
        MemoryObject *propertiesMo =
            memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);     
        ObjectState* os = bindObjectInState(state, propertiesMo, false);
        os->write(4, ZExtExpr::create(symmo->scalarType, Expr::Int8));
        bindLocal(ki, state, propertiesMo->getBaseExpr());   
        return;  
      } else {
        terminateStateOnProgramError(state, "not found tensor symblic array", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("TensorIteratorBase5dtype")!=std::string::npos) {
      int index = 0;
      if (auto indexCE = dyn_cast<ConstantExpr>(arguments[1])) {
        index = indexCE->getZExtValue();
      } else {
        terminateStateOnProgramError(state, "TensorIteratorBase5dtype: index is not constant", StateTerminationType::ReportError);
        return;
      }

      std::string symName = state.inputIterator->getTensor(index);
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
      llvm::Type *returnType = f->getReturnType();
      unsigned width = returnType->getIntegerBitWidth();

      if (!symmo->scalarType) {
        std::string typeName = symName + ".scalar_type";
        MemoryObject *typeMo =
              memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                              &state, /*allocSite=*/state.prevPC->inst,
                              /*alignment=*/8);          
        typeMo->setName(typeName);
        const Array *typeArray = arrayCache.CreateArray(typeName, typeMo->size, true);
        bindObjectInState(state, typeMo, false, typeArray);
        state.addSymbolic(typeMo, typeArray);
        ref<Expr> scalarTypeExpr = Expr::createTempRead(typeArray, typeMo->size*8);
        addConstraint(state, UgeExpr::create(scalarTypeExpr, ConstantExpr::create(0, scalarTypeExpr->getWidth())));
        addConstraint(state, UleExpr::create(scalarTypeExpr, ConstantExpr::create(255, scalarTypeExpr->getWidth())));
        symmo->scalarType = scalarTypeExpr;
      }
      bindLocal(ki, state, ZExtExpr::create(symmo->scalarType, width));
      return;
    } else if (f->getName().find("TensorIteratorBase11input_dtype")!=std::string::npos) {
      int index = 0;
      if (auto indexCE = dyn_cast<ConstantExpr>(arguments[1])) {
        index = indexCE->getZExtValue();
      } else {
        terminateStateOnProgramError(state, "TensorIteratorBase5dtype: index is not constant", StateTerminationType::ReportError);
        return;
      }
      index += state.inputIterator->noutputs;

      if (index >= state.inputIterator->tensors.size()) {
        terminateStateOnProgramError(state, "TensorIteratorBase5dtype: index is larger than tensors.size()", StateTerminationType::ReportError);
        return;
      }

      std::string symName = state.inputIterator->getTensor(index);
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
      llvm::Type *returnType = f->getReturnType();
      unsigned width = returnType->getIntegerBitWidth();

      if (!symmo->scalarType) {
        std::string typeName = symName + ".scalar_type";
        MemoryObject *typeMo =
              memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                              &state, /*allocSite=*/state.prevPC->inst,
                              /*alignment=*/8);          
        typeMo->setName(typeName);
        const Array *typeArray = arrayCache.CreateArray(typeName, typeMo->size, true);
        bindObjectInState(state, typeMo, false, typeArray);
        state.addSymbolic(typeMo, typeArray);
        ref<Expr> scalarTypeExpr = Expr::createTempRead(typeArray, typeMo->size*8);
        addConstraint(state, UgeExpr::create(scalarTypeExpr, ConstantExpr::create(0, scalarTypeExpr->getWidth())));
        addConstraint(state, UleExpr::create(scalarTypeExpr, ConstantExpr::create(255, scalarTypeExpr->getWidth())));
        symmo->scalarType = scalarTypeExpr;
      }
      bindLocal(ki, state, ZExtExpr::create(symmo->scalarType, width));
      return;
    } else if (f->getName().find("TensorBase13is_contiguous")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("13is_contiguous")!=std::string::npos && f->getName().find("Tensor")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorBase6device")!=std::string::npos) {
      bindLocal(ki, state, ConcatExpr::create(ConstantExpr::create(1, Expr::Int8), ConstantExpr::create(1, Expr::Int8)));
      return;
    } else if (f->getName().find("TensorIteratorBase6device")!=std::string::npos) {
      bindLocal(ki, state, ConcatExpr::create(ConstantExpr::create(1, Expr::Int8), ConstantExpr::create(1, Expr::Int8)));
      return;
    } else if (f->getName().find("TensorBase6layout")!=std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("torch5empty")!=std::string::npos || f->getName().find("5torch5zeros")!=std::string::npos 
              || f->getName().find("at5zeros")!=std::string::npos || f->getName().find("at5empty")!=std::string::npos
              || f->getName().find("at7randintEll")!=std::string::npos || f->getName().find("at4ones")!=std::string::npos) {
      std::string funcName = f->getName().str();
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "torch::"+funcName+": do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      SymArrayMemoryObject *symmo;
      if (state.symbolicArrayMap.find(symName) == state.symbolicArrayMap.end()) {
        unsigned id = 0;
        std::string uniqueName = "empty_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "empty_tensor_" + llvm::utostr(++id);
        }
        mo->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
        bindObjectInState(state, mo, false, array);
        state.addSymbolic(mo, array);

        symmo = new SymArrayMemoryObject(mo->address, array, nullptr, nullptr);
        ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
        uint64_t destAddress = destAddressCE->getZExtValue();
        if (destAddress != mo->address) {
          symmo->arrayAddress = destAddress;
        }
        state.symbolicArrayMap[mo->name] = symmo;
      } else {
        symmo = state.symbolicArrayMap[symName];
      }

      if (funcName.find("zeros") != std::string::npos) {
        symmo->minVal = ConstantExpr::create(0, Expr::Int32);
        symmo->maxVal = ConstantExpr::create(0, Expr::Int32);
      } else if (funcName.find("ones") != std::string::npos) {
        symmo->minVal = ConstantExpr::create(1, Expr::Int32);
        symmo->maxVal = ConstantExpr::create(1, Expr::Int32);
      }
      
      if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        ObjectPair op2;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op2, success) || !success) {
          terminateStateOnProgramError(state, "torch::"+funcName+": do not find the shape array object", StateTerminationType::ReportError);
          return;
        }

        const ObjectState *shapeArrayOS = op2.second;
        ref<Expr> sizeExpr = nullptr;
        for (uint64_t i = 0; i < shapeSize; ++i) {
            ref<Expr> element = shapeArrayOS->read(i*8, 64);
            if (i == 0) {
              sizeExpr = element;
            } else {
              sizeExpr = MulExpr::create(sizeExpr, element);
            }
          symmo->dimensionSize[i] = element;
        }
        symmo->size = sizeExpr;
        symmo->shapeSize = arguments[2];

        if (arguments[3]->getWidth() == 64) {
          ref<Expr> scalarType = ExtractExpr::create(arguments[3], 16, 16);
          symmo->scalarType = scalarType;
          unsigned size = getTensorTypeBytes(symmo->scalarType);
          if (size > 0)
            symmo->elementSize = ConstantExpr::create(size, Expr::Int64);
          else {
            bool setItemSize = false;
            ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
            if (readExpr) {
              std::string readName = readExpr->updates.root->name;
              std::string suffix = ".scalar_type";
              if (readName.size() >= suffix.size()) {
                if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
                  std::string tensorName = readName.substr(0, readName.size() - suffix.size());
                  if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                    SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                    if (originalSymmo->elementSize) {
                      symmo->elementSize = originalSymmo->elementSize;
                      setItemSize = true;
                    }
                  }
                }
              }
            }
            if (!setItemSize) {
              std::string elementSizeName = mo->name + ".item_size";
              auto pair2 = createSizeSymbol(state, elementSizeName);
              ref<Expr> elementSizeExpr = pair2.second;
              symmo->elementSize = elementSizeExpr;
              addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
              addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
            }
          }
        } else if (arguments[3]->getWidth() == 16) {
          ref<Expr> scalarType = ExtractExpr::create(arguments[3], 0, 8);
          symmo->scalarType = scalarType;
        }
        return;
      } else {
        terminateStateOnProgramError(state, "torch::"+funcName+": the size of the shape array is not a constant", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("torch10empty_like")!=std::string::npos || f->getName().find("at10zeros_like")!=std::string::npos
              || f->getName().find("2at6Tensor3mulERKN3c106ScalarE")!=std::string::npos || f->getName().find("2at6Tensor3subERKN3c106ScalarE")!=std::string::npos
            || f->getName().find("2at10empty_like")!=std::string::npos || f->getName().find("at9ones_like")!=std::string::npos) {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "torch::empty_like: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "torch::empty_like: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
        SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "empty tensor", "torch_empty_like_tensor_");
        
        destSymmo->size = sourceSymmo->size;
        destSymmo->sizeName = sourceSymmo->sizeName;
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->scalarType = sourceSymmo->scalarType;
        destSymmo->elementSize = sourceSymmo->elementSize;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
        return;
      } else {
        terminateStateOnProgramError(state, "torch::empty_like: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("2at6Tensor5zero_Ev")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "at::zeros_like: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *sourceMO = op0.first;
      std::string symName = getSymName(state, sourceMO, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[symName];
        MemoryObject *mo =
            memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);    
        unsigned id = 0;
        std::string uniqueName = "at_zerod_like_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "at_zerod_like_tensor_" + llvm::utostr(++id);
        }
        mo->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, mo->size, false);
        bindObjectInState(state, mo, false, array);
        state.addSymbolic(mo, array);

        SymArrayMemoryObject *destSymmo = new SymArrayMemoryObject(mo->address, array, nullptr, sourceSymmo->size);
        state.symbolicArrayMap[mo->name] = destSymmo;
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->scalarType = sourceSymmo->scalarType;
        destSymmo->elementSize = sourceSymmo->elementSize;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
        bindLocal(ki, state, mo->getBaseExpr());
        return;
      } else {
        terminateStateOnProgramError(state, "at::zeros_like: do not find symbolic tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at15empty_quantizedEN3c108ArrayRef")!=std::string::npos) {
      bool success;
      ObjectPair op3;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[3], op3, success) || !success) {
        terminateStateOnProgramError(state, "at::empty_quantized: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "at::empty_quantized: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op3.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[3]);
      const MemoryObject *destMO = op0.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
        SymArrayMemoryObject *destSymmo;
        ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
        uint64_t destAddress = destAddressCE->getZExtValue();
        if (destAddress != destMO->address) {
          if (state.symNames.find(destAddress)!=state.symNames.end()) {
            std::string destMName = state.symNames[destAddress];
            if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
              terminateStateOnProgramError(state, "at::empty_quantized: do not find target tensor object", StateTerminationType::ReportError);
              return;
            }
            destSymmo = state.symbolicArrayMap[destMName];
          } else {
            unsigned id = 0;
            std::string uniqueName = "torch_empty_quantized_tensor_0";
            while (!state.arrayNames.insert(uniqueName).second) {
              uniqueName = "torch_empty_quantized_tensor_" + llvm::utostr(++id);
            }
            const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
            if (sourceSymmo->size) {
              destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, sourceSymmo->size);
            } else {
              std::string sizeName = uniqueName + ".size";
              auto pair = createSizeSymbol(state, sizeName, true);
              ref<Expr> sizeExpr = pair.second;
              destSymmo = new SymArrayMemoryObject(destAddress, array, pair.first, sizeExpr);
            }
            state.symbolicArrayMap[uniqueName] = destSymmo;
            state.symNames[destAddress] = uniqueName;
            state.symAddressMap[destMO->address] = destAddress;
          }
        } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
          unsigned id = 0;
          std::string uniqueName = "torch_empty_quantized_tensor_0";
          while (!state.arrayNames.insert(uniqueName).second) {
            uniqueName = "torch_empty_quantized_tensor_" + llvm::utostr(++id);
          }
          destMO->setName(uniqueName);
          const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
          bindObjectInState(state, destMO, false, array);
          state.addSymbolic(destMO, array);

          std::string sizeName = destMO->name + ".size";
          auto pair = createSizeSymbol(state, sizeName, true);
          ref<Expr> sizeExpr = pair.second;
          destSymmo = new SymArrayMemoryObject(destMO->address, array, pair.first, sizeExpr);
          state.symbolicArrayMap[destMO->name] = destSymmo;
        } else {
          destSymmo = state.symbolicArrayMap[destMO->name];
        }

        if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
          uint64_t shapeSize = shapeSizeExpr->getZExtValue();
          ObjectPair op1;
          if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
            terminateStateOnProgramError(state, "empty_quantized: do not find the shape array object", StateTerminationType::ReportError);
            return;
          }
  
          const ObjectState *shapeArrayOS = op1.second;
          ref<Expr> sizeExpr = nullptr;
          for (uint64_t i = 0; i < shapeSize; ++i) {
              ref<Expr> element = shapeArrayOS->read(i*8, 64);
              if (i == 0) {
                sizeExpr = element;
              } else {
                sizeExpr = MulExpr::create(sizeExpr, element);
              }
              destSymmo->dimensionSize[i] = element;
          }
          destSymmo->size = sizeExpr;
          destSymmo->shapeSize = arguments[2];
  
          ref<Expr> scalarType = ExtractExpr::create(arguments[4], 16, 16);
          destSymmo->scalarType = scalarType;
          unsigned size = getTensorTypeBytes(destSymmo->scalarType);
          if (size > 0)
            destSymmo->elementSize = ConstantExpr::create(size, Expr::Int64);
          else {
            bool setItemSize = false;
            ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
            if (readExpr) {
              std::string readName = readExpr->updates.root->name;
              std::string suffix = ".scalar_type";
              if (readName.size() >= suffix.size()) {
                if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
                  std::string tensorName = readName.substr(0, readName.size() - suffix.size());
                  if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                    SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                    if (originalSymmo->elementSize) {
                      destSymmo->elementSize = originalSymmo->elementSize;
                      setItemSize = true;
                    }
                  }
                }
              }
            }
            if (!setItemSize) {
              std::string elementSizeName = destMO->name + ".item_size";
              auto pair2 = createSizeSymbol(state, elementSizeName);
              ref<Expr> elementSizeExpr = pair2.second;
              destSymmo->elementSize = elementSizeExpr;
              addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
              addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
            }
          }
          destSymmo->isQuantized = ConstantExpr::create(1, Expr::Bool);
          destSymmo->qscheme = sourceSymmo->qscheme;
          state.symbolicArrayMap[destMO->name] = destSymmo;
          return;
        } else {
          klee_warning("empty_quantized: the size of the shape array is not a constant");
          destSymmo->shapeSize = arguments[2];
  
          ref<Expr> scalarType = ExtractExpr::create(arguments[4], 16, 16);
          destSymmo->scalarType = scalarType;
          unsigned size = getTensorTypeBytes(destSymmo->scalarType);
          if (size > 0)
            destSymmo->elementSize = ConstantExpr::create(size, Expr::Int64);
          else {
            bool setItemSize = false;
            ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
            if (readExpr) {
              std::string readName = readExpr->updates.root->name;
              std::string suffix = ".scalar_type";
              if (readName.size() >= suffix.size()) {
                if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
                  std::string tensorName = readName.substr(0, readName.size() - suffix.size());
                  if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                    SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                    if (originalSymmo->elementSize) {
                      destSymmo->elementSize = originalSymmo->elementSize;
                      setItemSize = true;
                    }
                  }
                }
              }
            }
            if (!setItemSize) {
              std::string elementSizeName = destMO->name + ".item_size";
              auto pair2 = createSizeSymbol(state, elementSizeName);
              ref<Expr> elementSizeExpr = pair2.second;
              destSymmo->elementSize = elementSizeExpr;
              addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
              addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
            }
          }
          destSymmo->isQuantized = ConstantExpr::create(1, Expr::Bool);
          destSymmo->qscheme = sourceSymmo->qscheme;
          state.symbolicArrayMap[destMO->name] = destSymmo;
          return;
        }
      } else {
        terminateStateOnProgramError(state, "empty_quantized: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at6Tensor5fill_ERKN3c106Scalar")!=std::string::npos) {
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("at6Tensor12q_zero_point")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "at::Tensor::q_zero_point: do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        llvm::Type *returnType = f->getReturnType();
        unsigned width = returnType->getIntegerBitWidth();
        if (symmo->zeroPoint.isNull()) {
          std::string pointName = symName + ".zero_point";
          MemoryObject *pointMo =
            memory->allocate(width/8, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
          ref<Expr> pointExpr = symbolizeIndex(state, pointName, pointMo, true);
          symmo->zeroPoint = pointExpr;
        }
        bindLocal(ki, state, ZExtExpr::create(symmo->zeroPoint, width));
        return;
      } else {
        terminateStateOnProgramError(state, "at::Tensor::q_zero_point: do not find symbolic tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at6Tensor4view")!=std::string::npos || f->getName().find("at6Tensor7reshape")!=std::string::npos
              || f->getName().find("2at8internal16expand_slow_pathERKNS_10TensorBaseEN3c108ArrayRef")!=std::string::npos) {
      std::string funcName;
      if (f->getName().find("at6Tensor4view")!=std::string::npos) {
        funcName = "view";
      } else if (f->getName().find("reshape")!=std::string::npos) {
        funcName = "reshape";
      } else {
        funcName = "expand_slow_path";
      }
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "torch::"+funcName+": do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "torch::"+funcName+": do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;

      if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[3])) {
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        ObjectPair op3;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op3, success) || !success) {
          terminateStateOnProgramError(state, "torch::"+funcName+": do not find the shape array object", StateTerminationType::ReportError);
          return;
        }

        if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
          SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
          SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "torch::"+funcName, funcName+"_tensor_");

          const ObjectState *shapeArrayOS = op3.second;
          int inferIndex = -1;
          ref<Expr> sizeExpr = ConstantExpr::create(1, 64);
          for (uint64_t i = 0; i < shapeSize; ++i) {
            ref<Expr> element = shapeArrayOS->read(i*8, 64);
            destSymmo->dimensionSize[i] = element;
            bool isNeg = false;

            if (auto CE = dyn_cast<ConstantExpr>(element)) {
              int64_t element_val = CE->getZExtValue();
              if (element_val == -1) {
                if (inferIndex >= 0) {
                  terminateStateOnProgramError(state, "torch::"+funcName+": shape multiple -1", StateTerminationType::ReportError);
                  return;
                }
                inferIndex = i;
                isNeg = true;
              }
            }

            if (!isNeg) {
              sizeExpr = MulExpr::create(sizeExpr, element);
            }
          }
          if (inferIndex >= 0) {
            destSymmo->dimensionSize[inferIndex] = UDivExpr::create(ZExtExpr::create(sourceSymmo->size, sizeExpr->getWidth()), sizeExpr);
            sizeExpr = sourceSymmo->size;
          }
          destSymmo->size = sizeExpr;
          destSymmo->shapeSize = arguments[3];
          destSymmo->scalarType = sourceSymmo->scalarType;
          destSymmo->elementSize = sourceSymmo->elementSize;
          destSymmo->maxVal = sourceSymmo->maxVal;
          destSymmo->minVal = sourceSymmo->minVal;
          destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          destSymmo->isFloat = sourceSymmo->isFloat;
          return;
        } else {
          terminateStateOnProgramError(state, "torch::"+funcName+": do not find target tensor object", StateTerminationType::ReportError);
          return;
        }
      } else {
        if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
          SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
          SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "torch::"+funcName, funcName+"_tensor_");
          destSymmo->size = sourceSymmo->size;
          destSymmo->shapeSize = arguments[3];
          destSymmo->scalarType = sourceSymmo->scalarType;
          destSymmo->elementSize = sourceSymmo->elementSize;
          destSymmo->maxVal = sourceSymmo->maxVal;
          destSymmo->minVal = sourceSymmo->minVal;
          destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          destSymmo->isFloat = sourceSymmo->isFloat;

          klee_warning("torch::%s: the size of the shape array is not a constant", funcName.c_str());
        } else {
          terminateStateOnProgramError(state, "torch::"+funcName+": do not find target tensor object", StateTerminationType::ReportError);
          return;
        }
        return;
      }
    } else if (f->getName().find("at6Tensor7flatten")!=std::string::npos) {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "torch::flatten: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "torch::flatten: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        if (isa<ConstantExpr>(arguments[2]) && isa<ConstantExpr>(arguments[3])) {
          auto startDimCE = dyn_cast<ConstantExpr>(arguments[2]);
          auto endDimCE = dyn_cast<ConstantExpr>(arguments[3]);
          int startDim = startDimCE->getZExtValue();
          int endDim = endDimCE->getZExtValue();

          SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
          SymArrayMemoryObject *destSymmo;
          ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
          uint64_t destAddress = destAddressCE->getZExtValue();
          if (destAddress != destMO->address) {
            if (state.symNames.find(destAddress)!=state.symNames.end()) {
              std::string destMName = state.symNames[destAddress];
              if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
                terminateStateOnProgramError(state, "torch::flatten: do not find target tensor object", StateTerminationType::ReportError);
                return;
              }
              destSymmo = state.symbolicArrayMap[destMName];
            } else {
              unsigned id = 0;
              std::string uniqueName = "torch_flatten_tensor_0";
              while (!state.arrayNames.insert(uniqueName).second) {
                uniqueName = "torch_flatten_tensor_" + llvm::utostr(++id);
              }
              const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
              destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
              state.symbolicArrayMap[uniqueName] = destSymmo;
              state.symNames[destAddress] = uniqueName;
              state.symAddressMap[destMO->address] = destAddress;
            }
          } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
            unsigned id = 0;
            std::string uniqueName = "torch_flatten_tensor_0";
            while (!state.arrayNames.insert(uniqueName).second) {
              uniqueName = "torch_flatten_tensor_" + llvm::utostr(++id);
            }
            destMO->setName(uniqueName);
            const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
            bindObjectInState(state, destMO, false, array);
            state.addSymbolic(destMO, array);
            destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
            state.symbolicArrayMap[destMO->name] = destSymmo;
          } else {
            destSymmo = state.symbolicArrayMap[destMO->name];
          }
          destSymmo->size = sourceSymmo->size;
          destSymmo->scalarType = sourceSymmo->scalarType;
          destSymmo->elementSize = sourceSymmo->elementSize;
          destSymmo->maxVal = sourceSymmo->maxVal;
          destSymmo->minVal = sourceSymmo->minVal;
          destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          destSymmo->isFloat = sourceSymmo->isFloat;

          if (startDim == 0 && endDim == -1) {            
            destSymmo->shapeSize = ConstantExpr::create(1, Expr::Int64);
            std::map<int, ref<Expr>> dimensionSize;
            std::map<int, ref<Expr>> strides;
            dimensionSize[0] = sourceSymmo->size;
            strides[0] = ConstantExpr::create(1, Expr::Int64);
            destSymmo->dimensionSize = dimensionSize;
            destSymmo->strides = strides;
            return;
          } else if (endDim == -1 && !sourceSymmo->shapeSize.isNull() && !isa<ConstantExpr>(sourceSymmo->shapeSize)) {
            destSymmo->shapeSize = ConstantExpr::create(startDim, sourceSymmo->shapeSize->getWidth());
            if (!sourceSymmo->dimensionSize.empty()) {
              ref<Expr> newDimSize = nullptr;
              for (auto &pair : sourceSymmo->dimensionSize) {
                int index = pair.first;
                if (pair.second.isNull()) {
                  continue;
                }
                if (index < startDim) {
                  destSymmo->dimensionSize[index] = pair.second;
                } else {
                  if (newDimSize) {
                    newDimSize = MulExpr::create(newDimSize, pair.second);
                  } else {
                    newDimSize = pair.second;
                  }
                }
              }
              if (!newDimSize.isNull())
                destSymmo->dimensionSize[startDim] = newDimSize;
            }
            return;
          } else{
            if (endDim == -1 && !sourceSymmo->shapeSize.isNull() && isa<ConstantExpr>(sourceSymmo->shapeSize)) {
              auto shapeSizeCE = dyn_cast<ConstantExpr>(sourceSymmo->shapeSize);
              int shapeSize = shapeSizeCE->getZExtValue();
              endDim = shapeSize-1;
            }
            if (!sourceSymmo->shapeSize.isNull()) {
              destSymmo->shapeSize = SubExpr::create(sourceSymmo->shapeSize, ConstantExpr::create(endDim-startDim, sourceSymmo->shapeSize->getWidth()));
            }
            if (!sourceSymmo->dimensionSize.empty()) {
              ref<Expr> newDimSize = nullptr;
              for (auto &pair : sourceSymmo->dimensionSize) {
                int index = pair.first;
                if (pair.second.isNull()) {
                  continue;
                }
                if (index < startDim) {
                  destSymmo->dimensionSize[index] = pair.second;
                } else if (index > endDim) {
                  destSymmo->dimensionSize[index-(endDim-startDim)] = pair.second;
                } else {
                  if (newDimSize) {
                    newDimSize = MulExpr::create(newDimSize, pair.second);
                  } else {
                    newDimSize = pair.second;
                  }
                }
              }
              if (!newDimSize.isNull())
                destSymmo->dimensionSize[startDim] = newDimSize;
            }
            std::map<int, ref<Expr>> strides;
            destSymmo->strides = strides;
            return;
          }
        } else {
          terminateStateOnProgramError(state, "torch::flatten: dimension not constant", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "torch::flatten: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at6Tensor9unsqueeze")!=std::string::npos) {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "torch::unsqueeze: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "torch::unsqueeze: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        if (auto dimCE = dyn_cast<ConstantExpr>(arguments[2])) {
          int dim = dimCE->getZExtValue();
          SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
          SymArrayMemoryObject *destSymmo;
          ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
          uint64_t destAddress = destAddressCE->getZExtValue();
          if (destAddress != destMO->address) {
            if (state.symNames.find(destAddress)!=state.symNames.end()) {
              std::string destMName = state.symNames[destAddress];
              if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
                terminateStateOnProgramError(state, "torch::unsqueeze: do not find target tensor object", StateTerminationType::ReportError);
        return;
              }
              destSymmo = state.symbolicArrayMap[destMName];
            } else {
              unsigned id = 0;
              std::string uniqueName = "torch_unsqueeze_tensor_0";
              while (!state.arrayNames.insert(uniqueName).second) {
                uniqueName = "torch_unsqueeze_tensor_" + llvm::utostr(++id);
              }
              const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
              destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
              state.symbolicArrayMap[uniqueName] = destSymmo;
              state.symNames[destAddress] = uniqueName;
              state.symAddressMap[destMO->address] = destAddress;
            }
          } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
            unsigned id = 0;
            std::string uniqueName = "torch_unsqueeze_tensor_0";
            while (!state.arrayNames.insert(uniqueName).second) {
              uniqueName = "torch_unsqueeze_tensor_" + llvm::utostr(++id);
            }
            destMO->setName(uniqueName);
            const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
            bindObjectInState(state, destMO, false, array);
            state.addSymbolic(destMO, array);
            destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
            state.symbolicArrayMap[destMO->name] = destSymmo;
          } else {
            destSymmo = state.symbolicArrayMap[destMO->name];
          }

          destSymmo->size = sourceSymmo->size;
          if (!sourceSymmo->shapeSize.isNull())
            destSymmo->shapeSize = AddExpr::create(sourceSymmo->shapeSize, ConstantExpr::create(1, sourceSymmo->shapeSize->getWidth()));
          destSymmo->scalarType = sourceSymmo->scalarType;
          destSymmo->elementSize = sourceSymmo->elementSize;
          destSymmo->maxVal = sourceSymmo->maxVal;
          destSymmo->minVal = sourceSymmo->minVal;
          destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          destSymmo->isFloat = sourceSymmo->isFloat;

          std::map<int, ref<Expr>> dimensionSize;
          std::map<int, ref<Expr>> strides;
          dimensionSize[dim] = ConstantExpr::create(1, Expr::Int64);
          for (auto &pair : sourceSymmo->dimensionSize) {
            int index = pair.first;
            if (index >= dim) {
              dimensionSize[index+1] = pair.second;
            } else {
              dimensionSize[index] = pair.second;
            }
          }
          destSymmo->dimensionSize = dimensionSize;
          destSymmo->strides = strides;
          return;
        } else {
          terminateStateOnProgramError(state, "torch::unsqueeze: dimension not constant", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "torch::unsqueeze: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("14expand_inplaceERKNS_6Tensor")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "expand_inplace: do not find return object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "expand_inplace: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success) {
        terminateStateOnProgramError(state, "expand_inplace: do not find size tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      std::string destSymName = getSymName(state, destMO, arguments[2]);
      if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
         terminateStateOnProgramError(state, "expand_inplace: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }
      if (state.symbolicArrayMap.find(destSymName) == state.symbolicArrayMap.end()) {
         terminateStateOnProgramError(state, "expand_inplace: do not find size tensor symbolic object", StateTerminationType::ReportError);
        return;
      }
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo = state.symbolicArrayMap[destSymName];

      unsigned id = 0;
      std::string uniqueName = "expand_inplace_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "expand_inplace_tensor_" + llvm::utostr(++id);
      }

      MemoryObject *returnMo =
            memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.pc->inst,
                            /*alignment=*/8);    
      returnMo->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, returnMo->size);
      bindObjectInState(state, returnMo, false, array);
      state.addSymbolic(returnMo, array);
      
      SymArrayMemoryObject *returnSymmo = new SymArrayMemoryObject(returnMo->address, array, destSymmo->sizeName, destSymmo->size);
      state.symbolicArrayMap[uniqueName] = returnSymmo;

      returnSymmo->shapeSize = destSymmo->shapeSize;
      returnSymmo->dimensionSize = destSymmo->dimensionSize;
      returnSymmo->strides = destSymmo->strides;
      returnSymmo->scalarType = sourceSymmo->scalarType;
      returnSymmo->elementSize = sourceSymmo->elementSize;
      returnSymmo->isQuantized = sourceSymmo->isQuantized;
      returnSymmo->qscheme = sourceSymmo->qscheme;
      returnSymmo->zeroPoint = sourceSymmo->zeroPoint;
      returnSymmo->maxVal = sourceSymmo->maxVal;
      returnSymmo->minVal = sourceSymmo->minVal;
      returnSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      returnSymmo->isFloat = sourceSymmo->isFloat;

      ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
      wos->write(0, ConstantExpr::create(1, Expr::Int8));
      wos->write(1, returnMo->getBaseExpr());
      return;
    } else if (f->getName().find("2at6Tensor9expand_as")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "2at6Tensor9expand_as: do not find return object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "2at6Tensor9expand_as: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success) {
        terminateStateOnProgramError(state, "2at6Tensor9expand_as: do not find size tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      const MemoryObject *destMO = op2.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      std::string destSymName = getSymName(state, destMO, arguments[2]);
      if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
        terminateStateOnProgramError(state, "expand_inplace: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }
      if (state.symbolicArrayMap.find(destSymName) == state.symbolicArrayMap.end()) {
        terminateStateOnProgramError(state, "expand_inplace: do not find size tensor symbolic object", StateTerminationType::ReportError);
        return;
      }
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo = state.symbolicArrayMap[destSymName];

      const MemoryObject *returnMO = op0.first;
      SymArrayMemoryObject *returnSymmo;
      if (state.symbolicArrayMap.find(returnMO->name) == state.symbolicArrayMap.end()) {
        unsigned id = 0;
        std::string uniqueName = "expand_as_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "expand_as_tensor_" + llvm::utostr(++id);
        }
        returnMO->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, returnMO->size);
        bindObjectInState(state, returnMO, false, array);
        state.addSymbolic(returnMO, array);

        returnSymmo = new SymArrayMemoryObject(returnMO->address, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = returnSymmo;
      } else {
        returnSymmo = state.symbolicArrayMap[returnMO->name];
      }

      returnSymmo->sizeName = destSymmo->sizeName;
      returnSymmo->size = destSymmo->size;
      returnSymmo->shapeSize = destSymmo->shapeSize;
      returnSymmo->dimensionSize = destSymmo->dimensionSize;
      returnSymmo->strides = destSymmo->strides;
      returnSymmo->scalarType = sourceSymmo->scalarType;
      returnSymmo->elementSize = sourceSymmo->elementSize;
      returnSymmo->isQuantized = sourceSymmo->isQuantized;
      returnSymmo->qscheme = sourceSymmo->qscheme;
      returnSymmo->zeroPoint = sourceSymmo->zeroPoint;
      returnSymmo->maxVal = sourceSymmo->maxVal;
      returnSymmo->minVal = sourceSymmo->minVal;
      returnSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      returnSymmo->isFloat = sourceSymmo->isFloat;
      return;
    } else if (f->getName().find("at6Tensor3sum")!=std::string::npos) {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "torch::sum: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "torch::sum: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op3;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op3, success) || !success) {
        terminateStateOnProgramError(state, "torch::view: do not find the shape array object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
        SymArrayMemoryObject *destSymmo;
        const ObjectState *shapeArrayOS = op3.second;
        ref<Expr> dimIndexExpr = shapeArrayOS->read(0, 64);
        if (auto dimIndexCE = dyn_cast<ConstantExpr>(dimIndexExpr)) {
          int dimIndex = dimIndexCE->getZExtValue();
          ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
          uint64_t destAddress = destAddressCE->getZExtValue();
          if (destAddress != destMO->address) {
            if (state.symNames.find(destAddress)!=state.symNames.end()) {
              std::string destMName = state.symNames[destAddress];
              if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
                terminateStateOnProgramError(state, "torch::sum: do not find target tensor object", StateTerminationType::ReportError);
                return;
              }
              destSymmo = state.symbolicArrayMap[destMName];
            } else {
              std::string uniqueName = sourceSymName+".sum("+llvm::utostr(dimIndex)+")";
              const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
              destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
              state.symbolicArrayMap[uniqueName] = destSymmo;
              state.symNames[destAddress] = uniqueName;
              state.symAddressMap[destMO->address] = destAddress;
            }
          } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
            std::string uniqueName = sourceSymName+".sum("+llvm::utostr(dimIndex)+")";
            destMO->setName(uniqueName);
            const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
            bindObjectInState(state, destMO, false, array);
            state.addSymbolic(destMO, array);
            destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
            state.symbolicArrayMap[destMO->name] = destSymmo;
          } else {
            destSymmo = state.symbolicArrayMap[destMO->name];
          }

          for (const auto& [key, value] : sourceSymmo->dimensionSize) {
            destSymmo->dimensionSize[key] = value;
          }
          
          destSymmo->dimensionSize[dimIndex] = ConstantExpr::create(1, 64);
          if (dimIndex==0 && !sourceSymmo->shapeSize.isNull()) {
            destSymmo->shapeSize = SubExpr::create(sourceSymmo->shapeSize, ConstantExpr::create(1, sourceSymmo->shapeSize->getWidth()));
          } else if (!sourceSymmo->shapeSize.isNull()) {
            destSymmo->shapeSize = sourceSymmo->shapeSize;
          }
          destSymmo->scalarType = sourceSymmo->scalarType;
          destSymmo->elementSize = sourceSymmo->elementSize;
          destSymmo->maxVal = sourceSymmo->maxVal;
          destSymmo->minVal = sourceSymmo->minVal;
          destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          destSymmo->isFloat = sourceSymmo->isFloat;
          return;
        } else {
          terminateStateOnProgramError(state, "torch::sum: dim is not constant", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "torch::sum: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("2at6TensorixEl")!=std::string::npos || f->getName().find("2at6Tensor6selectEl")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "tensor get element: do not find target object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "tensor get element: do not find source object", StateTerminationType::ReportError);
        return;
      }

      ref<Expr> indexExpr = nullptr;
      if (arguments.size() == 4) {
        indexExpr = arguments[3];
      } else {
        indexExpr = arguments[2];
      }

      const MemoryObject *sourceMo = op1.first;
      const MemoryObject *destMo = op0.first;
      if (state.symbolicArrayMap.find(sourceMo->name) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceMo->name];
        if (state.symbolicArrayMap.find(destMo->name) == state.symbolicArrayMap.end()) {
          std::string elementName;
          if (auto indexExprCE = dyn_cast<ConstantExpr>(indexExpr)) {
            int index = indexExprCE->getZExtValue();
            elementName = sourceMo->name+"["+llvm::utostr(index)+"]";
            sourceSymmo->constantIndexAddress[index] = destMo->getBaseExpr();
          } else {
            unsigned id = 0;
            elementName = sourceMo->name+"[symbol0]";
            while (!state.arrayNames.insert(elementName).second) {
              elementName = sourceMo->name+"[symbol" + llvm::utostr(++id) +"]";
            }
            destMo->setName(elementName);
            sourceSymmo->symbolicIndexAddress[indexExpr] = destMo->getBaseExpr();
          }
          destMo->setName(elementName);
          const Array *array = arrayCache.CreateArray(elementName, destMo->size);
          bindObjectInState(state, destMo, false, array);
          state.addSymbolic(destMo, array);

          ref<Expr> dim0Size;
          if (!sourceSymmo->dimensionSize.empty() && sourceSymmo->dimensionSize.find(0) != sourceSymmo->dimensionSize.end()) {
            dim0Size = sourceSymmo->dimensionSize[0];
          } else {
            std::string dim0SizeName = sourceMo->name + ".size[0]";
            auto dim0SizePair = createSizeSymbol(state, dim0SizeName);
            ref<Expr> dim0SizeExpr = dim0SizePair.second; 
            addConstraint(state, UgeExpr::create(dim0SizeExpr, ConstantExpr::create(1, dim0SizeExpr->getWidth())));  
            sourceSymmo->dimensionSize[0] = dim0SizeExpr;
            dim0Size = dim0SizeExpr;
          }
          ref<Expr> totalSize = UDivExpr::create(sourceSymmo->size, ZExtExpr::create(dim0Size, sourceSymmo->size->getWidth()));

          SymArrayMemoryObject *elementSymmo = new SymArrayMemoryObject(destMo->address, array, nullptr, totalSize);
          elementSymmo->scalarType = sourceSymmo->scalarType;
          elementSymmo->elementSize = sourceSymmo->elementSize;
          elementSymmo->shapeSize = SubExpr::create(sourceSymmo->shapeSize, ConstantExpr::create(1, sourceSymmo->shapeSize->getWidth()));
          elementSymmo->maxVal = sourceSymmo->maxVal;
          elementSymmo->minVal = sourceSymmo->minVal;
          elementSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          elementSymmo->isFloat = sourceSymmo->isFloat;
          state.symbolicArrayMap[elementName] = elementSymmo;

          if (!sourceSymmo->dimensionSize.empty()) {
            for (const auto& [key, value] : sourceSymmo->dimensionSize) {
              if (key > 0)
                elementSymmo->dimensionSize[key-1] = value;
            }
          }
        } else {
          if (auto indexExprCE = dyn_cast<ConstantExpr>(indexExpr)) {
            int index = indexExprCE->getZExtValue();
            sourceSymmo->constantIndexAddress[index] = destMo->getBaseExpr();
          } else {
            sourceSymmo->symbolicIndexAddress[indexExpr] = destMo->getBaseExpr();
          }
        }
        return;
      } else {
        terminateStateOnProgramError(state, "tensor get element: source object is not symbolic", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("_ZN2at6TensorD2Ev")!=std::string::npos || f->getName().find("_ZN3c104cuda9CUDAGuardD2Ev")!=std::string::npos
              || f->getName().find("_ZN3c107SymBoolD2Ev")!=std::string::npos || f->getName().find("_ZN3c106SymIntD2Ev")!=std::string::npos
            || f->getName().find("_ZN3c104cuda9CUDAGuardC2Ea")!=std::string::npos) {
      return;
    } else if (f->getName().find("at6TensorC2Ev")!=std::string::npos) {
      ObjectPair op;
      bool success;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "tensor constructor: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      SymArrayMemoryObject *symmo = createSymArray(state, arguments[0], mo, "tensor constructor", "tensor_");

      std::string sizeName = symmo->arrayName->name + ".size";
      auto pair = createSizeSymbol(state, sizeName, true);
      symmo->sizeName = pair.first;
      symmo->size = pair.second;
      
      std::string elementSizeName = mo->name + ".item_size";
      auto pair2 = createSizeSymbol(state, elementSizeName);
      ref<Expr> elementSizeExpr = pair2.second;
      symmo->elementSize = elementSizeExpr;
      addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
      addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
      return;
    } else if (f->getName().find("2at6TensorC2ERKS0_")!=std::string::npos || f->getName().find("2at6Tensor2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE")!=std::string::npos 
              || f->getName().find("2at6TensorC2EOS0_")!=std::string::npos || f->getName().find("2at6TensoraSEOS0_")!=std::string::npos
              || f->getName().find("at6TensorC2ENS_10TensorBase15unsafe_borrow")!=std::string::npos
              || f->getName().find("2at6Tensor10contiguousEN3c1012MemoryFormatE")!=std::string::npos
              || f->getName().find("2at6Tensor5cloneE")!=std::string::npos || f->getName().find("2at6Tensor5copy_E")!=std::string::npos
              || f->getName().find("at6Tensor10resize_as_ERKS0_St8optionalIN3c1012MemoryFormat")!=std::string::npos) {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "tensor copy: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "tensor copy: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
        SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "tensor copy", "copy_tensor_");
        destSymmo->size = sourceSymmo->size;
        destSymmo->sizeName = sourceSymmo->sizeName;
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->scalarType = sourceSymmo->scalarType;
        destSymmo->elementSize = sourceSymmo->elementSize;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
        destSymmo->maxVal = sourceSymmo->maxVal;
        destSymmo->minVal = sourceSymmo->minVal;
        destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
        destSymmo->isFloat = sourceSymmo->isFloat;
        
        llvm::Type *returnType = f->getReturnType();
        if (returnType->isPointerTy()) {
          llvm::StructType *structType = dyn_cast<llvm::StructType>(returnType->getPointerElementType());
          if (structType && structType->getName().find("at::Tensor")!=std::string::npos) {
            bindLocal(ki, state, arguments[0]);
          }
        }
        return;
      } else {
        terminateStateOnProgramError(state, "tensor copy: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName() == "_ZNR2at6TensoraSERKS0_") {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "tensor operator=: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "tensor operator=: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
        SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "tensor copy", "assign_tensor_");
        destSymmo->size = sourceSymmo->size;
        destSymmo->sizeName = sourceSymmo->sizeName;
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->scalarType = sourceSymmo->scalarType;
        destSymmo->elementSize = sourceSymmo->elementSize;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
        destSymmo->maxVal = sourceSymmo->maxVal;
        destSymmo->minVal = sourceSymmo->minVal;
        destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
        destSymmo->isFloat = sourceSymmo->isFloat;
        bindLocal(ki, state, arguments[0]);
        return;
      } else {
        terminateStateOnProgramError(state, "tensor operator=: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at6Tensor2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE")!=std::string::npos) {
      return;
    } else if (f->getName().find("8optionalIN2at9GeneratorEE9has_value")!=std::string::npos) {
      if (isa<ConstantExpr>(arguments[0])) {
        auto CE = dyn_cast<ConstantExpr>(arguments[0]);
        int address = CE->getZExtValue();
        if (address == 0) {
          bindLocal(ki, state, ConstantExpr::create(0, Expr::Bool));
          return;
        }
      }

      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "optional<Generator>.has_value(): do not find target object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        bindLocal(ki, state, ConstantExpr::create(0, Expr::Bool));
        return;
      }
    } else if (f->getName().find("8optional")!=std::string::npos && (f->getName().find("9has_value")!=std::string::npos || f->getName().find("IN2at6TensorEEcvbEv")!=std::string::npos)) {
      if (isa<ConstantExpr>(arguments[0])) {
        auto CE = dyn_cast<ConstantExpr>(arguments[0]);
        int address = CE->getZExtValue();
        if (address == 0) {
          bindLocal(ki, state, ConstantExpr::create(0, Expr::Bool));
          return;
        }
      }
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "optional.has_value(): do not find target object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        if ((!symmo->shapeSize.isNull() && isa<ConstantExpr>(symmo->shapeSize)) || !symmo->dimensionSize.empty()) {
          bindLocal(ki, state, ConstantExpr::create(1, Expr::Bool));
        } else {
          MemoryObject *boolMo =
                memory->allocate(1, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
          std::string boolName = symName + ".has_value";
          boolMo->setName(boolName);
          const Array *boolArray = arrayCache.CreateArray(boolName, boolMo->size, true);
          bindObjectInState(state, boolMo, false, boolArray);
          state.addSymbolic(boolMo, boolArray);
          ref<Expr> boolExpr = Expr::createTempRead(boolArray, boolMo->size*8);
          addConstraint(state, UgeExpr::create(boolExpr, ConstantExpr::create(0, Expr::Int8)));
          addConstraint(state, UleExpr::create(boolExpr, ConstantExpr::create(1, Expr::Int8)));
          ref<Expr> boolCond = NeExpr::create(boolExpr, ConstantExpr::create(0, Expr::Int8)); 
          bindLocal(ki, state, boolCond);
        }
        return;
      }
    } else if (f->getName().find("8optional")!=std::string::npos && (f->getName().find("5value")!=std::string::npos || f->getName().find("2at6TensorEEptEv")!=std::string::npos)) {
      if (isa<ConstantExpr>(arguments[0])) {
        auto CE = dyn_cast<ConstantExpr>(arguments[0]);
        int address = CE->getZExtValue();
        if (address == 0) {
          bindLocal(ki, state, arguments[0]);
          return;
        }
      }

      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "optional.value(): do not find target object", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        if (f->getName().find("2at6TensorEE")!=std::string::npos) {
          SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
          setElementSize(state, symmo);

          if (!symmo->elementSize) {
            std::string elementSizeName = symName + ".item_size";
            auto pair = createSizeSymbol(state, elementSizeName);
            ref<Expr> elementSizeExpr = pair.second;
            symmo->elementSize = elementSizeExpr;
            addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
            addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
          }
        }
        bindLocal(ki, state, arguments[0]);
        return;
      }
    } else if (f->getName().find("8optionalIN2at6TensorEEC2ERKS2_")!=std::string::npos) {
      bool success;
      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
        terminateStateOnProgramError(state, "8optionalIN2at6TensorEEC2ERKS2_: do not find target optional<torch::tensor> object", StateTerminationType::ReportError);
        return;
      }

      if (isa<ConstantExpr>(arguments[1])) {
        auto CE = dyn_cast<ConstantExpr>(arguments[1]);
        int address = CE->getZExtValue();
        if (address == 0) {
          ObjectState *wos = state.addressSpace.getWriteable(op2.first, op2.second);
          wos->write(8, ConstantExpr::create(0, Expr::Int8));
          return;
        }
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "8optionalIN2at6TensorEEC2ERKS2_: do not find source optional<torch::tensor> object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op2.first;
      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
        SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "tensor copy", "assign_tensor_");
        
        destSymmo->size = sourceSymmo->size;
        destSymmo->sizeName = sourceSymmo->sizeName;
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->scalarType = sourceSymmo->scalarType;
        destSymmo->elementSize = sourceSymmo->elementSize;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
        destSymmo->maxVal = sourceSymmo->maxVal;
        destSymmo->minVal = sourceSymmo->minVal;
        destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
        destSymmo->isFloat = sourceSymmo->isFloat;
        bindLocal(ki, state, arguments[0]);
        return;
      } else {
        terminateStateOnProgramError(state, "8optionalIN2at6TensorEEC2ERKS2_: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }
    }


    else if (f->getName().find("8optionalIN2at6TensorEED2Ev")!=std::string::npos) {
      return;
    } else if (f->getName().find("5torch9from_blob")!=std::string::npos || f->getName().find("2at9from_blob")!=std::string::npos) { // TODO: copy data from args[1]
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      SymArrayMemoryObject *symmo = createSymArray(state, arguments[0], mo, "from_blob", "from_blob_tensor_");

      
      if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[3])) {
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        ObjectPair op2;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success) {
          terminateStateOnProgramError(state, "torch::from_blob: do not find the shape array object", StateTerminationType::ReportError);
        return;
        }

        const ObjectState *shapeArrayOS = op2.second;
        ref<Expr> sizeExpr = nullptr;
        for (uint64_t i = 0; i < shapeSize; ++i) {
            ref<Expr> element = shapeArrayOS->read(i*8, 64);
            if (i == 0) {
              sizeExpr = element;
            } else {
              sizeExpr = MulExpr::create(sizeExpr, element);
            }
          symmo->dimensionSize[i] = element;
        }
        symmo->size = sizeExpr;
        symmo->shapeSize = arguments[3];

        ObjectPair op4;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[4], op4, success) || !success) {
          terminateStateOnProgramError(state, "torch::from_blob: do not find the tensoroption object", StateTerminationType::ReportError);
        return;
        }
        const ObjectState *optionOS = op4.second;
        ref<Expr> scalarType = optionOS->read(2, 16);
        symmo->scalarType = scalarType;
        unsigned size = getTensorTypeBytes(symmo->scalarType);
        if (size > 0)
          symmo->elementSize = ConstantExpr::create(size, Expr::Int64);
        else {
          bool setItemSize = false;
          ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
          if (readExpr) {
            std::string readName = readExpr->updates.root->name;
            std::string suffix = ".scalar_type";
            if (readName.size() >= suffix.size()) {
              if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
                std::string tensorName = readName.substr(0, readName.size() - suffix.size());
                if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                  SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                  if (originalSymmo->elementSize) {
                    symmo->elementSize = originalSymmo->elementSize;
                    setItemSize = true;
                  }
                }
              }
            }
          }
          if (!setItemSize) {
            std::string elementSizeName = symmo->arrayName->name + ".item_size";
            auto pair2 = createSizeSymbol(state, elementSizeName);
            ref<Expr> elementSizeExpr = pair2.second;
            symmo->elementSize = elementSizeExpr;
            addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
            addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
          }
        }
        return;
      } else {
        terminateStateOnProgramError(state, "torch::from_blob: the size of the shape array is not a constant", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("at6Tensor4itemEv")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "at6Tensor4itemEv: do not find scalar object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "at6Tensor4itemEv: do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
      const MemoryObject *tensorMo = op1.first;

      std::string typeName = tensorMo->name + ".tag";
      MemoryObject *typeMo =
            memory->allocate(4, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
      typeMo->setName(typeName);
      const Array *typeArray = arrayCache.CreateArray(typeName, typeMo->size, true);
      bindObjectInState(state, typeMo, false, typeArray);
      state.addSymbolic(typeMo, typeArray);
      ref<Expr> scalarTypeExpr = Expr::createTempRead(typeArray, typeMo->size*8);
      addConstraint(state, UgeExpr::create(scalarTypeExpr, ConstantExpr::create(0, scalarTypeExpr->getWidth())));
      addConstraint(state, UleExpr::create(scalarTypeExpr, ConstantExpr::create(4, scalarTypeExpr->getWidth()))); // not support symoblic value now
      wos->write(0, scalarTypeExpr);

      std::string double1Name = tensorMo->name + "_double1";
      MemoryObject *d1Mo =
        memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                        &state, /*allocSite=*/state.prevPC->inst,
                        /*alignment=*/8);          
      ref<Expr> d1Expr = symbolizeIndex(state, double1Name, d1Mo, true);
      wos->write(16, d1Expr);

      std::string double2Name = tensorMo->name + "_double2";
      MemoryObject *d2Mo =
        memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                        &state, /*allocSite=*/state.prevPC->inst,
                        /*alignment=*/8);          
      ref<Expr> d2Expr = symbolizeIndex(state, double2Name, d2Mo, true);
      wos->write(24, d1Expr);
      return;
    } else if (f->getName().find("6vector")!=std::string::npos && f->getName().find("4size")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "vector.size(): do not find vector object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        llvm::Type *returnType = f->getReturnType();
        bindLocal(ki, state, ZExtExpr::create(state.symbolicArrayMap[symName]->size, returnType->getIntegerBitWidth()));
        return;
      }
    } else if (f->getName().find("6vectorIN2at6TensorESaIS1_EEixEm")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "vector get element: do not find vector object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        std::string elementName;
        bool isConstantIndex = false;
        int index;
        if (auto indexExpr = dyn_cast<ConstantExpr>(arguments[1])) {
          index = indexExpr->getZExtValue();
          isConstantIndex = true;
          if (symmo->constantIndexAddress.find(index) == symmo->constantIndexAddress.end()) {
            elementName = symName+"["+llvm::utostr(index)+"]";
          }
        } else {
          if (symmo->symbolicIndexAddress.find(arguments[1]) == symmo->symbolicIndexAddress.end()) {
            unsigned id = 0;
            elementName = symName + "[symbol0]";
            while (!state.arrayNames.insert(elementName).second) {
              elementName = symName + "[symbol" + llvm::utostr(++id) +"]";
            }
          }
        }

        if (!elementName.empty()) {
          MemoryObject *elementMo =
            memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);    
          elementMo->setName(elementName);
          const Array *array = arrayCache.CreateArray(elementName, mo->size);
          bindObjectInState(state, elementMo, false, array);
          state.addSymbolic(elementMo, array);

          std::string sizeName = elementName + ".size";
          auto pair = createSizeSymbol(state, sizeName, true);
          ref<Expr> sizeExpr = pair.second;
          SymArrayMemoryObject *elementSymmo = new SymArrayMemoryObject(elementMo->address, array, pair.first, sizeExpr);
          std::string dimName = elementName + ".dim";
          auto dimpair = createSizeSymbol(state, dimName);
          ref<Expr> dimExpr = dimpair.second; 
          elementSymmo->shapeSize = dimExpr;

          std::string itemSizeName = elementName + ".item_size";
          auto pair2 = createSizeSymbol(state, itemSizeName);
          ref<Expr> itemSizeExpr = pair2.second;
          elementSymmo->elementSize = itemSizeExpr;
          addConstraint(state, UgeExpr::create(itemSizeExpr, ConstantExpr::create(1, itemSizeExpr->getWidth())));
          addConstraint(state, UleExpr::create(itemSizeExpr, ConstantExpr::create(16, itemSizeExpr->getWidth())));
          state.symbolicArrayMap[elementName] = elementSymmo;

          if (isConstantIndex) {
            symmo->constantIndexAddress[index] = elementMo->getBaseExpr();
          } else {
            symmo->symbolicIndexAddress[arguments[1]] = elementMo->getBaseExpr();
          }
        }
        if (isConstantIndex) {
          bindLocal(ki, state, symmo->constantIndexAddress[index]);
        } else {
          bindLocal(ki, state, symmo->symbolicIndexAddress[arguments[1]]);
        }
        return;
      } else {
        klee_warning("vector get element: vector object is not symbolic");
      }
    } else if (f->getName().find("6vectorIlSaIlEEixEm")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "vector get element: do not find vector object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end() || state.tensorSizesMap.find(symName) != state.tensorSizesMap.end()) {
        llvm::Type *returnType = f->getReturnType();
        if (returnType->isPointerTy()) {
          llvm::Type *elementType = returnType->getPointerElementType();
          unsigned width = getTypeBits(elementType);
          if (width == 0) {
            terminateStateOnProgramError(state, "vector get element: element type not handled", StateTerminationType::ReportError);
            return;
          }
          ref<Expr> result = AddExpr::create(mo->getBaseExpr(), MulExpr::create(arguments[1], ConstantExpr::create(width/8, arguments[1]->getWidth())));
          state.base_mos[mo->address].insert(result);
          state.base_addrs[result] = mo->getBaseExpr();
          SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
          if (symmo->elementSize.isNull()) {
            symmo->elementSize = ConstantExpr::create(width/8, Expr::Int64);
          }
          bindLocal(ki, state, result);
          return;
        } else {
          terminateStateOnProgramError(state, "vector get element: do not return pointer", StateTerminationType::ReportError);
        return;
        }
      } else {
        klee_warning("vector get element: vector object is not symbolic");
      }
    } else if (f->getName().find("6vectorIlSaIlEE8pop_backEv")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "vector pop_back(): do not find vector object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.tensorSizesMap.find(symName) != state.tensorSizesMap.end()) {
        TensorSizesMemoryObject *sizesSymmo = state.tensorSizesMap[symName];
        if (!sizesSymmo->elements.empty() && !sizesSymmo->length.isNull()) {
          if (auto lengthCE = dyn_cast<ConstantExpr>(sizesSymmo->length)) {
            int length = lengthCE->getZExtValue();
            sizesSymmo->elements.erase(length-1);
            sizesSymmo->length = ConstantExpr::create(length-1, lengthCE->getWidth());
            SymArrayMemoryObject *symmo = state.symbolicArrayMap[sizesSymmo->tensorName];
            symmo->dimensionSize.erase(length-1);
            symmo->shapeSize = sizesSymmo->length;
            return;
          }
        }
      }
    } else if (f->getName().find("6vectorIlSaIlEE9push_backEOl")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "vector push_back(): do not find vector object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "vector push_back(): do not find pushed object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo0 = op0.first;
      std::string symName = getSymName(state, mo0, arguments[0]);
      if (state.tensorSizesMap.find(symName) != state.tensorSizesMap.end()) {
        TensorSizesMemoryObject *sizesSymmo = state.tensorSizesMap[symName];
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[sizesSymmo->tensorName];
        const ObjectState *os = op1.second;
        ref<Expr> value = os->read(0, Expr::Int64);

        if (!sizesSymmo->elements.empty() && !sizesSymmo->length.isNull()) {
          if (auto lengthCE = dyn_cast<ConstantExpr>(sizesSymmo->length)) {
            int length = lengthCE->getZExtValue();
            sizesSymmo->elements[length] = value;
            sizesSymmo->length = ConstantExpr::create(length+1, lengthCE->getWidth());
            symmo->dimensionSize[length] = value;
            symmo->shapeSize = sizesSymmo->length;
            return;
          } else {
            sizesSymmo->elements[-1] = value;
            symmo->dimensionSize[-1] = value;
            return;
          }
        } else {
          sizesSymmo->elements[-1] = value;
          symmo->dimensionSize[-1] = value;
          return;
        }
      }
    } else if (f->getName() == "_ZNKSt6vectorIlSaIlEE5beginEv") {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "vector.begin(): do not find vector object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        bindLocal(ki, state, arguments[0]);
        return;
      } else {
        klee_warning("vector.begin(): vector object is not symbolic");
      }
    } else if (f->getName() == "_ZNKSt6vectorIlSaIlEE3endEv") {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "vector.end(): do not find vector object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        bindLocal(ki, state, AddExpr::create(arguments[0], ZExtExpr::create(symmo->size, Expr::Int64)));
        return;
      } else {
        klee_warning("vector.end(): vector object is not symbolic");
      }
    } else if (f->getName().find("_ZSt10accumulateIN9__gnu_cxx17__normal_iteratorIPKlSt6vectorIlSaIlEEEEiET0_T_S9_S8_")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "accumulate vector: do not find begin object", StateTerminationType::ReportError);
        return;
      } 
      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
        if (isa<ConstantExpr>(SubExpr::create(arguments[1], arguments[0]))) {
          ref<ConstantExpr> size_CE = dyn_cast<ConstantExpr>(SubExpr::create(arguments[1], arguments[0]));
          int size = size_CE->getZExtValue();
          const ObjectState *os = op.second;
          ref<Expr> sumExpr = ConstantExpr::create(0, Expr::Int64);
          for (int i = 0; i < size; i++) {
            ref<Expr> val = os->read(i*8, Expr::Int64);
            sumExpr = AddExpr::create(sumExpr, val);
          }
          bindLocal(ki, state, sumExpr);
        } else {
          std::string sumName = symName + ".sum";
          MemoryObject *sumMo =
            memory->allocate(sizeof(unsigned), /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
          ref<Expr> sumExpr = symbolizeIndex(state, sumName, sumMo);
          bindLocal(ki, state, sumExpr);
        }
        return;
      } else {
        klee_warning("accumulate vector: vector object is not symbolic");
      }
    } else if (f->getName() == "_ZNK3c108ArrayRefIlE3vecEv") {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
        terminateStateOnProgramError(state, "_ZNK3c108ArrayRefIlE3vecEv: do not find arrayref pointer", StateTerminationType::ReportError);
        return;
      }
      ref<Expr> address = op.second->read(0, 64);
      ref<Expr> size = op.second->read(8, 64);
      if (!isa<ConstantExpr>(size)) {
        ObjectPair op2;
        if (!state.addressSpace.resolveOne(state, solver.get(), address, op2, success) || !success) {
          terminateStateOnProgramError(state, "_ZNK3c108ArrayRefIlE3vecEv: do not find array object", StateTerminationType::ReportError);
          return;
        }

        ObjectPair op0;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
          terminateStateOnProgramError(state, "_ZNK3c108ArrayRefIlE3vecEv: do not find vector pointer", StateTerminationType::ReportError);
          return;
        }

        const MemoryObject *mo = op2.first;
        if (state.tensorSizesMap.find(mo->name) != state.tensorSizesMap.end()) {
          TensorSizesMemoryObject* tensorSizeSymo = state.tensorSizesMap[mo->name];
          std::string vecName = mo->name + ".vec()";
          const MemoryObject *vecMo = op0.first;     
          vecMo->setName(vecName);
          const Array *vecArray = arrayCache.CreateArray(vecName, vecMo->size);
          bindObjectInState(state, vecMo, false, vecArray);
          state.addSymbolic(vecMo, vecArray);
          TensorSizesMemoryObject* vecSymMO = new TensorSizesMemoryObject(vecMo->address, vecArray, tensorSizeSymo->tensorName, tensorSizeSymo->length);
          SymArrayMemoryObject* symmo = state.symbolicArrayMap[tensorSizeSymo->tensorName];
          if (mo->name.find("sizes")!=std::string::npos) {
            vecSymMO->elements = symmo->dimensionSize;
          } else if (mo->name.find("strides")!=std::string::npos) {
            vecSymMO->elements = symmo->strides;
          } else {
            vecSymMO->elements = tensorSizeSymo->elements;
          }
          state.tensorSizesMap[vecName] = vecSymMO;
          return;
        }
      }
    } else if (f->getName() == "_ZNSt6vectorIN3c106SymIntESaIS1_EEC2IPKlvEET_S7_RKS2_") {
      if (!isa<ConstantExpr>(arguments[2])) {
        bool success;
        ObjectPair op0;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
          terminateStateOnProgramError(state, "_ZNSt6vectorIN3c106SymIntESaIS1_EEC2IPKlvEET_S7_RKS2_: do not find vector pointer", StateTerminationType::ReportError);
          return;
        }

        ObjectPair op1;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
          terminateStateOnProgramError(state, "_ZNSt6vectorIN3c106SymIntESaIS1_EEC2IPKlvEET_S7_RKS2_: do not find begin pointer", StateTerminationType::ReportError);
        return;
        }

        const MemoryObject *mo = op1.first;
        if (state.tensorSizesMap.find(mo->name) != state.tensorSizesMap.end()) {
          TensorSizesMemoryObject* tensorSizeSymo = state.tensorSizesMap[mo->name];
          std::string vecName = mo->name + ".vec()";
          const MemoryObject *vecMo = op0.first;     
          vecMo->setName(vecName);
          const Array *vecArray = arrayCache.CreateArray(vecName, vecMo->size);
          bindObjectInState(state, vecMo, false, vecArray);
          state.addSymbolic(vecMo, vecArray);
          
          TensorSizesMemoryObject* vecSymMO = new TensorSizesMemoryObject(vecMo->address, vecArray, tensorSizeSymo->tensorName, tensorSizeSymo->length);
          SymArrayMemoryObject* symmo = state.symbolicArrayMap[tensorSizeSymo->tensorName];
          if (mo->name.find("sizes")!=std::string::npos) {
            vecSymMO->elements = symmo->dimensionSize;
          } else if (mo->name.find("strides")!=std::string::npos) {
            vecSymMO->elements = symmo->strides;
          }
          state.tensorSizesMap[vecName] = vecSymMO;
          return;
        }
      }
    } else if (f->getName() == "_ZN3c108ArrayRefIlEC2ISaIlEEERKSt6vectorIlT_E") {
      bool success;
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "_ZN3c108ArrayRefIlEC2ISaIlEEERKSt6vectorIlT_E: do not find vector pointer", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op1, success) || !success) {
        terminateStateOnProgramError(state, "_ZN3c108ArrayRefIlEC2ISaIlEEERKSt6vectorIlT_E: do not find array pointer", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo1 = op1.first;
      if (state.tensorSizesMap.find(mo1->name) != state.tensorSizesMap.end()) {
        TensorSizesMemoryObject* sizesSymMO = state.tensorSizesMap[mo1->name];
        ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
        wos->write(0, arguments[1]);
        wos->write(8, ZExtExpr::create(sizesSymMO->length, Expr::Int64));
        return;
      }
    } else if (f->getName() == "_ZN3c10eqIlEEbNS_8ArrayRefIT_EES3_") {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "_ZN3c10eqIlEEbNS_8ArrayRefIT_EES3_: do not find first array object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo0 = op0.first;
      if (state.tensorSizesMap.find(mo0->name)!=state.tensorSizesMap.end()) {
        ObjectPair op1;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op1, success) || !success) {
          terminateStateOnProgramError(state, "_ZN3c10eqIlEEbNS_8ArrayRefIT_EES3_: do not find second array object", StateTerminationType::ReportError);
          return;
        }

        TensorSizesMemoryObject *sizeMo0 = state.tensorSizesMap[mo0->name];
        SymArrayMemoryObject *symmo0 = state.symbolicArrayMap[sizeMo0->tensorName];
        if (auto shapeSizeCE = dyn_cast<ConstantExpr>(arguments[3])) {
          int shapeSize = shapeSizeCE->getZExtValue();
          if (auto dimCE = dyn_cast<ConstantExpr>(arguments[1])) {
            int dim = dimCE->getZExtValue();
            if (dim != shapeSize) {
              bindLocal(ki, state, ConstantExpr::create(0, Expr::Bool));
              return;
            }
          }
          const ObjectState *shapeArrayOS = op1.second;
          ref<Expr> sizeExpr = nullptr;
          bool setSize = false;
          for (int i = 0; i < shapeSize; ++i) {
            ref<Expr> element = shapeArrayOS->read(i*8, 64);
            if (i == 0) {
              sizeExpr = element;
            } else {
              sizeExpr = MulExpr::create(sizeExpr, element);
            }

            if (!symmo0->dimensionSize.empty() && symmo0->dimensionSize.find(i) != symmo0->dimensionSize.end()) {
              if (element == symmo0->dimensionSize[i]) continue;
              if (auto elementCE = dyn_cast<ConstantExpr>(element)) {
                if (auto dimSizeCE = dyn_cast<ConstantExpr>(symmo0->dimensionSize[i])) {
                  int elementVal = elementCE->getZExtValue();
                  int dimSize = dimSizeCE->getZExtValue();
                  if (dimSize != elementVal) {
                    bindLocal(ki, state, ConstantExpr::create(0, Expr::Bool));
                    return;
                  }
                } else {
                  addConstraint(state, EqExpr::create(element, ZExtExpr::create(symmo0->dimensionSize[i], Expr::Int64)));
                  setSize = true;
                  symmo0->dimensionSize[i] = element;
                }
              } else {
                addConstraint(state, EqExpr::create(element, ZExtExpr::create(symmo0->dimensionSize[i], Expr::Int64)));
                if (!isa<ConstantExpr>(symmo0->dimensionSize[i])) {
                  setSize = true;
                  symmo0->dimensionSize[i] = element;
                }
              }
            } else {
              setSize = true;
              symmo0->dimensionSize[i] = element;
            }
          }

          if (setSize) {
            symmo0->size = sizeExpr;
          }
          symmo0->shapeSize = arguments[3];
          bindLocal(ki, state, ConstantExpr::create(1, Expr::Bool));
          return;
        }
      }
    } else if (f->getName().find("c1024legacyExtractDispatchKeyENS_14DispatchKeySet") != std::string::npos || f->getName().find("c1020dispatchKeyToBackendENS_11DispatchKey") != std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("TensorBase21suggest_memory_format") != std::string::npos) {
      llvm::Type *returnType = f->getReturnType();
      bindLocal(ki, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
      return;
    } else if (f->getName().find("at20TensorIteratorConfigC2Ev") != std::string::npos) {
      if (!isa<ConstantExpr>(arguments[0])) {
        terminateStateOnProgramError(state, "TensorIterator address is not constant", StateTerminationType::ReportError);
        return;
      }
      auto aCE = dyn_cast<ConstantExpr>(arguments[0]);
      uint64_t address = aCE->getZExtValue();
      TensorIterator *iterator = new TensorIterator(address, nullptr, nullptr, nullptr);      
      state.inputIterator = iterator;
      return;
    } else if (f->getName().find("TensorIteratorConfig")!=std::string::npos && f->getName().find("10TensorBase")!=std::string::npos 
      && f->getName().find("add")!=std::string::npos && f->getName().find("const_input")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
        terminateStateOnProgramError(state, "TensorIteratorConfig::add_const_input: do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      state.inputIterator->addTensor(mo->name, true);
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("TensorIteratorConfig")!=std::string::npos && f->getName().find("10TensorBase")!=std::string::npos 
        && f->getName().find("add")!=std::string::npos && f->getName().find("_input")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
        terminateStateOnProgramError(state, "TensorIteratorConfig::add_const_input: do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      state.inputIterator->addTensor(mo->name, false);
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("TensorIteratorConfig")!=std::string::npos && f->getName().find("10TensorBase")!=std::string::npos 
        && f->getName().find("add")!=std::string::npos && f->getName().find("const_output")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
        terminateStateOnProgramError(state, "TensorIteratorConfig::add_output: do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      state.inputIterator->addTensor(mo->name, true);
      state.inputIterator->noutputs+=1;
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("TensorIteratorConfig")!=std::string::npos && f->getName().find("10TensorBase")!=std::string::npos 
        && f->getName().find("add")!=std::string::npos && f->getName().find("_output")!=std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
        terminateStateOnProgramError(state, "TensorIteratorConfig::add_output: do not find tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      state.inputIterator->addTensor(mo->name, false);
      state.inputIterator->noutputs+=1;
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("TensorIteratorConfig20check_all_same_dtype") != std::string::npos) {
      if (!isa<ConstantExpr>(arguments[1])) {
        terminateStateOnProgramError(state, "TensorIteratorConfig20check_all_same_dtype: bool is not a constant", StateTerminationType::ReportError);
        return;
      }

      auto CE = dyn_cast<ConstantExpr>(arguments[1]);
      bool checkAllSameDtype = CE->getZExtValue();
      state.inputIterator->checkAllSameDtype = checkAllSameDtype;
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("TensorIteratorConfig5build") != std::string::npos) {
      computeTensorIteratorShape(state);
      return;
    } else if (f->getName().find("6vectorIlSaIlEE17_S_check_init_lenEmRKS0_") != std::string::npos || f->getName().find("6vectorIN2at6TensorESaIS1_EE17_S_check_init_lenEmRKS2_") != std::string::npos
              || f->getName().find("6vectorIN3c106SymIntESaIS1_EE17_S_check_init_lenEmRKS2_") != std::string::npos) {
      if (!isa<ConstantExpr>(arguments[0])) {
        addConstraint(state, UleExpr::create(arguments[0], ConstantExpr::create(1152921504606846975ULL, Expr::Int64)));
        bindLocal(ki, state, arguments[0]);
        return;
      }
    } else if (f->getName().find("Tensor7resize_EN3c108ArrayRefIlEESt8optionalINS1_12MemoryFormat") != std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "at::tensor::resize: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *mo = op.first;
      std::string symName = getSymName(state, mo, arguments[0]);
      if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();

        SymArrayMemoryObject *symmo;
        ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
        uint64_t destAddress = destAddressCE->getZExtValue();
        if (destAddress != mo->address) {
          if (state.symNames.find(destAddress)!=state.symNames.end()) {
            std::string destMName = state.symNames[destAddress];
            if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
              terminateStateOnProgramError(state, "at::tensor::resize: do not find target tensor object", StateTerminationType::ReportError);
        return;
            }
            symmo = state.symbolicArrayMap[destMName];
          } else {
            unsigned id = 0;
            std::string uniqueName = "at_resize_tensor_0";
            while (!state.arrayNames.insert(uniqueName).second) {
              uniqueName = "at_resize_tensor_" + llvm::utostr(++id);
            }
            const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
            symmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
            state.symbolicArrayMap[uniqueName] = symmo;
            state.symNames[destAddress] = uniqueName;
            state.symAddressMap[mo->address] = destAddress;
          }
        } else if (state.symbolicArrayMap.find(mo->name) == state.symbolicArrayMap.end()) {
          unsigned id = 0;
          std::string uniqueName = "at_resize_tensor_0";
          while (!state.arrayNames.insert(uniqueName).second) {
            uniqueName = "at_resize_tensor_" + llvm::utostr(++id);
          }
          mo->setName(uniqueName);
          const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
          bindObjectInState(state, mo, false, array);
          state.addSymbolic(mo, array);
          symmo = new SymArrayMemoryObject(mo->address, array, nullptr, nullptr);
          state.symbolicArrayMap[mo->name] = symmo;
        } else {
          symmo = state.symbolicArrayMap[symName];
        }

        
        ref<Expr> sizeExpr = ConstantExpr::create(1, 64);
        ObjectPair op1;
        if (!arguments[1].isNull()) {
          if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
            klee_warning("at::tensor::resize: do not find source tensor object");
          } else {
            const ObjectState *shapeArrayOS = op1.second;
            for (uint64_t i = 0; i < shapeSize; ++i) {
              ref<Expr> element = shapeArrayOS->read(i*8, 64);
              symmo->dimensionSize[i] = element;
              sizeExpr = MulExpr::create(sizeExpr, element);
            }
          }
        }
        symmo->size = sizeExpr;
        symmo->shapeSize = arguments[2];

        setElementSize(state, symmo);
        if (!symmo->elementSize) {
          std::string elementSizeName = mo->name + ".item_size";
          auto pair2 = createSizeSymbol(state, elementSizeName);
          ref<Expr> elementSizeExpr = pair2.second;
          symmo->elementSize = elementSizeExpr;
          addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
          addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
        }
        bindLocal(ki, state, arguments[0]);
        return;
      } else {
        terminateStateOnProgramError(state, "at::tensor::resize: the size of the shape array is not a constant", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("Tensor11as_strided_EN3c108ArrayRefIlEES3_St8optional") != std::string::npos) {
      bool success;
      ObjectPair op;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
        terminateStateOnProgramError(state, "at::tensor::as_strided_: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      if (!isa<ConstantExpr>(arguments[2]) || !isa<ConstantExpr>(arguments[4])) {
        terminateStateOnProgramError(state, "at::tensor::as_strided_: shapeSize is not constant", StateTerminationType::ReportError);
        return;
      }
      const MemoryObject *sourceMo = op.first;
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceMo->name];

      unsigned id = 0;
      std::string uniqueName = "as_strided_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "as_strided_tensor_" + llvm::utostr(++id);
      }

      MemoryObject *destMo =
            memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);    
      destMo->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, destMo->size);
      bindObjectInState(state, destMo, false, array);
      state.addSymbolic(destMo, array);

      SymArrayMemoryObject *destSymmo = new SymArrayMemoryObject(destMo->address, array, nullptr, nullptr);
      state.symbolicArrayMap[destMo->name] = destSymmo;

      auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2]);
      uint64_t shapeSize = shapeSizeExpr->getZExtValue();
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "at::tensor::as_strided_: do not find the shape array object", StateTerminationType::ReportError);
        return;
      }    
      const ObjectState *shapeArrayOS = op1.second;
      ref<Expr> sizeExpr = ConstantExpr::create(1, 64);
      for (uint64_t i = 0; i < shapeSize; ++i) {
        ref<Expr> element = shapeArrayOS->read(i*8, 64);
        destSymmo->dimensionSize[i] = element;
        sizeExpr = MulExpr::create(sizeExpr, element);
      }
      destSymmo->size = sizeExpr;
      destSymmo->shapeSize = arguments[2];  

      auto strideSizeExpr = dyn_cast<ConstantExpr>(arguments[4]);
      uint64_t strideSize = strideSizeExpr->getZExtValue();
      ObjectPair op3;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[3], op3, success) || !success) {
        terminateStateOnProgramError(state, "at::tensor::as_strided_: do not find the stride array object", StateTerminationType::ReportError);
        return;
      }    
      const ObjectState *strideArrayOS = op3.second;
      for (uint64_t i = 0; i < strideSize; ++i) {
        ref<Expr> element = strideArrayOS->read(i*8, 64);
        destSymmo->strides[i] = element;
      }
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;

      bindLocal(ki, state, destMo->getBaseExpr());
      return;
    } else if (f->getName().find("20TensorIteratorConfig21set_check_mem_overlap") != std::string::npos) {
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("20TensorIteratorConfig14resize_outputs") != std::string::npos) {
      if (!isa<ConstantExpr>(arguments[1])) {
        terminateStateOnProgramError(state, "TensorIteratorConfig14resize_outputs: bool is not a constant", StateTerminationType::ReportError);
        return;
      }

      auto CE = dyn_cast<ConstantExpr>(arguments[1]);
      bool resizeOutput = CE->getZExtValue();
      state.inputIterator->resizeOutput = resizeOutput;
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("at7squeezeERKNS_6Tensor") != std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "at7squeezeERKNS_6Tensor: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "at7squeezeERKNS_6Tensor: do not find target tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op0.first;

      if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
        terminateStateOnProgramError(state, "at7squeezeERKNS_6Tensor: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }

      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo;
      ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
      uint64_t destAddress = destAddressCE->getZExtValue();
      if (destAddress != destMO->address) {
        if (state.symNames.find(destAddress)!=state.symNames.end()) {
          std::string destMName = state.symNames[destAddress];
          if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
            terminateStateOnProgramError(state, "at7squeezeERKNS_6Tensor: do not find target tensor object", StateTerminationType::ReportError);
        return;
          }
          destSymmo = state.symbolicArrayMap[destMName];
        } else {
          unsigned id = 0;
          std::string uniqueName = "torch_squeeze_tensor_0";
          while (!state.arrayNames.insert(uniqueName).second) {
            uniqueName = "torch_squeeze_tensor_" + llvm::utostr(++id);
          }
          const Array *array = arrayCache.CreateArray(uniqueName, sourceMO->size);
          destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
          state.symbolicArrayMap[uniqueName] = destSymmo;
          state.symNames[destAddress] = uniqueName;
          state.symAddressMap[destMO->address] = destAddress;
        }
      } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
        unsigned id = 0;
        std::string uniqueName = "torch_squeeze_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "torch_squeeze_tensor_" + llvm::utostr(++id);
        }
        destMO->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
        bindObjectInState(state, destMO, false, array);
        state.addSymbolic(destMO, array);
        destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
        state.symbolicArrayMap[destMO->name] = destSymmo;
      } else {
        destSymmo = state.symbolicArrayMap[destMO->name];
      }

      if (!sourceSymmo->shapeSize.isNull() && isa<ConstantExpr>(sourceSymmo->shapeSize)) {
        auto shapeSizeExpr = dyn_cast<ConstantExpr>(sourceSymmo->shapeSize);
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        int index = 0;
        for (uint64_t i = 0; i < shapeSize; ++i) {
          if (auto dimSizeExpr = dyn_cast<ConstantExpr>(sourceSymmo->dimensionSize[i])) {
            uint64_t dimSize = dimSizeExpr->getZExtValue();
            if (dimSize == 1) {
              continue;
            }
          }
          destSymmo->dimensionSize[index] = sourceSymmo->dimensionSize[i];
          index+=1;
        }
        destSymmo->shapeSize = ConstantExpr::create(index, Expr::Int64);
      } else {
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
      }
      destSymmo->sizeName = sourceSymmo->sizeName;
      destSymmo->size = sourceSymmo->size;
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->isQuantized = sourceSymmo->isQuantized;
      destSymmo->qscheme = sourceSymmo->qscheme;
      destSymmo->zeroPoint= sourceSymmo->zeroPoint;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;
      return;
    } else if (f->getName().find("2at18TensorIteratorBaseD2Ev") != std::string::npos || f->getName().find("at14TensorIteratorD2Ev") != std::string::npos) {
      return;
    } else if (f->getName().find("at6Tensor6cumsumElSt8optionalIN3c1010ScalarType") != std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "at6Tensor6cumsumElSt8optionalIN3c1010ScalarType: do not find return tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "at6Tensor6cumsumElSt8optionalIN3c1010ScalarType: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op0.first;

      if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
        terminateStateOnProgramError(state, "at6Tensor6cumsumElSt8optionalIN3c1010ScalarType: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }

      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo;
      ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
      uint64_t destAddress = destAddressCE->getZExtValue();
      if (destAddress != destMO->address) {
        if (state.symNames.find(destAddress)!=state.symNames.end()) {
          std::string destMName = state.symNames[destAddress];
          if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
            terminateStateOnProgramError(state, "at6Tensor6cumsumElSt8optionalIN3c1010ScalarType: do not find target tensor object", StateTerminationType::ReportError);
        return;
          }
          destSymmo = state.symbolicArrayMap[destMName];
        } else {
          unsigned id = 0;
          std::string uniqueName = "cumsum_tensor_0";
          while (!state.arrayNames.insert(uniqueName).second) {
            uniqueName = "cumsum_tensor_" + llvm::utostr(++id);
          }
          const Array *array = arrayCache.CreateArray(uniqueName, sourceMO->size);
          destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
          state.symbolicArrayMap[uniqueName] = destSymmo;
          state.symNames[destAddress] = uniqueName;
          state.symAddressMap[destMO->address] = destAddress;
        }
      } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
        unsigned id = 0;
        std::string uniqueName = "cumsum_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "cumsum_tensor_" + llvm::utostr(++id);
        }
        destMO->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
        bindObjectInState(state, destMO, false, array);
        state.addSymbolic(destMO, array);
        destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
        state.symbolicArrayMap[destMO->name] = destSymmo;
      } else {
        destSymmo = state.symbolicArrayMap[destMO->name];
      }

      destSymmo->dimensionSize = sourceSymmo->dimensionSize;
      destSymmo->strides = sourceSymmo->strides;
      destSymmo->shapeSize = sourceSymmo->shapeSize;
      destSymmo->sizeName = sourceSymmo->sizeName;
      destSymmo->size = sourceSymmo->size;
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->isQuantized = sourceSymmo->isQuantized;
      destSymmo->qscheme = sourceSymmo->qscheme;
      destSymmo->zeroPoint= sourceSymmo->zeroPoint;

      ref<Expr> scalarType = ExtractExpr::create(arguments[3], 0, 8);
      if (!scalarType.isNull()) {
        if (auto CE = dyn_cast<ConstantExpr>(scalarType)) {
          int sval = CE->getZExtValue();
          if (sval>0) {
            destSymmo->scalarType = scalarType;
          }
        }
      }
      return;
    } else if (f->getName().find("3c106SymIntaSERKS0_") != std::string::npos || f->getName().find("3c106SymIntC2ERKS0_") != std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "3c106SymIntaSERKS0_: do not find target object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "3c106SymIntaSERKS0_: do not find source object", StateTerminationType::ReportError);
        return;
      }

      const ObjectState *sourceOS = op1.second;
      ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
      ref<Expr> value = sourceOS->read(0, Expr::Int64);
      wos->write(0, value);
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("2at6Tensor9transposeEll")!=std::string::npos || f->getName().find("2at6Tensor1tEv")!=std::string::npos) { 
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "2at6Tensor9transposeEll: do not find return tensor object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "2at6Tensor9transposeEll: do not find source tensor object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
      const MemoryObject *destMO = op0.first;

      if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
        terminateStateOnProgramError(state, "2at6Tensor9transposeEll: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
      }

      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo;
      if (destMO->name.empty() || destMO->name == "unnamed" || state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
        unsigned id = 0;
        std::string uniqueName = "transpose_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "transpose_tensor_" + llvm::utostr(++id);
        }
        destMO->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
        bindObjectInState(state, destMO, false, array);
        state.addSymbolic(destMO, array);
        destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
        state.symbolicArrayMap[destMO->name] = destSymmo;
      } else {
        destSymmo = state.symbolicArrayMap[destMO->name];
      }
      destSymmo->shapeSize = sourceSymmo->shapeSize;
      destSymmo->sizeName = sourceSymmo->sizeName;
      destSymmo->size = sourceSymmo->size;
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->isQuantized = sourceSymmo->isQuantized;
      destSymmo->qscheme = sourceSymmo->qscheme;
      destSymmo->zeroPoint= sourceSymmo->zeroPoint;

      if (!sourceSymmo->shapeSize.isNull() && isa<ConstantExpr>(sourceSymmo->shapeSize)) {
        auto shapeSizeExpr = dyn_cast<ConstantExpr>(sourceSymmo->shapeSize);
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        if (shapeSize == 2) {
          ref<Expr> dim0 = sourceSymmo->dimensionSize[0];
          ref<Expr> dim1 = sourceSymmo->dimensionSize[1];
          destSymmo->dimensionSize[0] = dim1;
          destSymmo->dimensionSize[1] = dim0;
          return;
        }
      }
    } else if (f->getName().find("TensorIteratorBase7strides")!=std::string::npos) {
      if (auto CE = dyn_cast<ConstantExpr>(arguments[1])) {
        int index = CE->getZExtValue();
        std::string name = state.inputIterator->tensors[index];
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[name];
        computeTensorIteratorStrides(state, symmo);

        int shapeSize = symmo->strides_bytes.size();
        MemoryObject *sizesMo =
              memory->allocate(shapeSize*8, /*isLocal=*/false, /*isGlobal=*/false,
                              &state, /*allocSite=*/state.prevPC->inst,
                              /*alignment=*/8);
        ObjectState *os = bindObjectInState(state, sizesMo, false);  
        for (int i = 0; i < shapeSize; i++) {
          os->write(i*8, ZExtExpr::create(symmo->strides_bytes[i], Expr::Int64));
        }
        ref<Expr> structExpr = ConcatExpr::create(ConstantExpr::create(shapeSize, Expr::Int64), ConstantExpr::create(sizesMo->address, 64));
        bindLocal(ki, state, structExpr);
        return;
      } else {
        terminateStateOnProgramError(state, "TensorIteratorBase7strides: index is not constant", StateTerminationType::ReportError);
        return;
      }
    } else if (f->getName().find("TensorIteratorBase5shape")!=std::string::npos) {
      computeTensorIteratorShape(state);
      int shapeSize = state.inputIterator->shape.size();
      MemoryObject *sizesMo =
            memory->allocate(shapeSize*8, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);
      ObjectState *os = bindObjectInState(state, sizesMo, false);  
      for (int i = 0; i < shapeSize; i++) {
        os->write(i*8, ZExtExpr::create(state.inputIterator->shape[i], Expr::Int64));
      }
      ref<Expr> structExpr = ConcatExpr::create(ConstantExpr::create(shapeSize, Expr::Int64), ConstantExpr::create(sizesMo->address, 64));
      bindLocal(ki, state, structExpr);
      return;
    } else if (f->getName().find("10TensorBase23unsafeReleaseTensorImplEv")!=std::string::npos) {
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("at10TensorBase28is_non_overlapping_and_dense")!=std::string::npos) {
      bindLocal(ki, state, ConstantExpr::create(1, Expr::Bool));
      return;
    } else if (f->getName().find("10TensorBase7defined")!=std::string::npos) {
      bindLocal(ki, state, ConstantExpr::create(1, Expr::Bool));
      return;
    } else if (f->getName().find("8optionalIN2at9GeneratorEEC2ERKS2_")!=std::string::npos) {
      bool success;
      ObjectPair op0;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
        terminateStateOnProgramError(state, "8optionalIN2at9GeneratorEEC2ERKS2_: do not find source object", StateTerminationType::ReportError);
        return;
      }

      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "8optionalIN2at9GeneratorEEC2ERKS2_: do not find target object", StateTerminationType::ReportError);
        return;
      }

      const MemoryObject *sourceMO = op1.first;
      std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);

      if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {

        const MemoryObject *destMO = op0.first;
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];

        unsigned id = 0;
        std::string uniqueName = "optional_copy_generator_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "optional_copy_generator_" + llvm::utostr(++id);
        }
        destMO->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
        bindObjectInState(state, destMO, false, array);
        state.addSymbolic(destMO, array);
        SymArrayMemoryObject *destSymmo = new SymArrayMemoryObject(destMO->address, array, sourceSymmo->sizeName, sourceSymmo->size);
        state.symbolicArrayMap[destMO->name] = destSymmo;
        return;
      }
    } else if (f->getName().find("6vectorIN3c106SymIntESaIS1_EED2Ev")!=std::string::npos) {
      return;
    } else if (f->getName().find("TensorBase11is_alias_ofERKS0_")!=std::string::npos) {
      bindLocal(ki, state, ConstantExpr::create(0, Expr::Bool));
      return;
    } else if (f->getName().find("3c10lsIlEERSoS1_NS_8ArrayRefIT_EE")!=std::string::npos) {
      bindLocal(ki, state, arguments[0]);
      return;
    } else if (f->getName().find("5equalIPKlS1_EbT_S2_T0_")!=std::string::npos) {
      if (!isa<ConstantExpr>(arguments[1])) {
        bool success;
        ObjectPair op0;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
          terminateStateOnProgramError(state, "5equalIPKlS1_EbT_S2_T0_: do not find first array object", StateTerminationType::ReportError);
        return;
        }

        ObjectPair op1;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op1, success) || !success) {
          terminateStateOnProgramError(state, "5equalIPKlS1_EbT_S2_T0_: do not find second array object", StateTerminationType::ReportError);
        return;
        }

        const MemoryObject *mo0 = op0.first;
        std::string symName0 = getSymName(state, mo0, arguments[0]);
        const MemoryObject *mo1 = op1.first;
        std::string symName1 = getSymName(state, mo1, arguments[2]);

        MemoryObject *boolMo =
                memory->allocate(1, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
        std::string boolName = symName0 + "_equals_" + symName1;
        boolMo->setName(boolName);
        const Array *boolArray = arrayCache.CreateArray(boolName, boolMo->size, true);
        bindObjectInState(state, boolMo, false, boolArray);
        state.addSymbolic(boolMo, boolArray);
        ref<Expr> boolExpr = Expr::createTempRead(boolArray, boolMo->size*8);
        addConstraint(state, UgeExpr::create(boolExpr, ConstantExpr::create(0, Expr::Int8)));
        addConstraint(state, UleExpr::create(boolExpr, ConstantExpr::create(1, Expr::Int8)));
        ref<Expr> boolCond = NeExpr::create(boolExpr, ConstantExpr::create(0, Expr::Int8)); 
        bindLocal(ki, state, boolCond);
        return;
      }
    }



          
          

    //   // Extract fields

    //   // Expand `signed_raw` from 1 bit to 8 bits





    // Check if maximum stack size was reached.
    // We currently only count the number of stack frames
    if (RuntimeMaxStackFrames && state.stack.size() > RuntimeMaxStackFrames) {
      terminateStateEarly(state, "Maximum stack size reached.", StateTerminationType::OutOfStackMemory);
      klee_warning("Maximum stack size reached.");
      return;
    }
    
    llvm::Type *retType = f->getReturnType();
    if (retType->isVoidTy()) {
      specialFunctionHandler->handleOpenMerge(state, ki, arguments);
    }

    // FIXME: I'm not really happy about this reliance on prevPC but it is ok, I
    // guess. This just done to avoid having to pass KInstIterator everywhere
    // instead of the actual instruction, since we can't make a KInstIterator
    KFunction *kf = kmodule->functionMap[f];

    state.pushFrame(state.prevPC, kf);
    state.pc = kf->instructions;

    if (statsTracker)
      statsTracker->framePushed(state, &state.stack[state.stack.size() - 2]);

    // TODO: support zeroext, signext, sret attributes

    unsigned callingArgs = arguments.size();
    unsigned funcArgs = f->arg_size();
    if (!f->isVarArg()) {
      if (callingArgs > funcArgs) {
        klee_warning_once(f, "calling %s with extra arguments.",
                          f->getName().data());
      } else if (callingArgs < funcArgs) {
        terminateStateOnUserError(state, "calling function with too few arguments");
        return;
      }
    } else {
      if (callingArgs < funcArgs) {
        terminateStateOnUserError(state, "calling function with too few arguments");
        return;
      }

      // Only x86-32 and x86-64 are supported
      Expr::Width WordSize = Context::get().getPointerWidth();
      assert(((WordSize == Expr::Int32) || (WordSize == Expr::Int64)) &&
             "Unknown word size!");

      uint64_t size = 0; // total size of variadic arguments
      bool requires16ByteAlignment = false;

      uint64_t offsets[callingArgs]; // offsets of variadic arguments
      uint64_t argWidth;             // width of current variadic argument

      const CallBase &cb = cast<CallBase>(*i);
      for (unsigned k = funcArgs; k < callingArgs; k++) {
        if (cb.isByValArgument(k)) {
          Type *t = cb.getParamByValType(k);
          argWidth = kmodule->targetData->getTypeSizeInBits(t);
        } else {
          argWidth = arguments[k]->getWidth();
        }

        MaybeAlign ma = cb.getParamAlign(k);
        unsigned alignment = ma ? ma->value() : 0;

        if (WordSize == Expr::Int32 && !alignment)
          alignment = 4;
        else {
          // 16 byte boundary if alignment needed by type exceeds 8 byte
          // boundary.
          if (!alignment && argWidth > Expr::Int64) {
            alignment = 16;
            requires16ByteAlignment = true;
          }

          if (!alignment)
            alignment = 8;
        }

        size = llvm::alignTo(size, alignment);
        offsets[k] = size;

        if (WordSize == Expr::Int32)
          size += Expr::getMinBytesForWidth(argWidth);
        else {
          size += llvm::alignTo(argWidth, WordSize) / 8;
        }
      }

      StackFrame &sf = state.stack.back();
      MemoryObject *mo = sf.varargs =
          memory->allocate(size, true, false, &state, state.prevPC->inst,
                           (requires16ByteAlignment ? 16 : 8));
      if (!mo && size) {
        terminateStateOnExecError(state, "out of memory (varargs)");
        return;
      }

      if (mo) {
        if ((WordSize == Expr::Int64) && (mo->address & 15) &&
            requires16ByteAlignment) {
          // Both 64bit Linux/Glibc and 64bit MacOSX should align to 16 bytes.
          klee_warning_once(
              0, "While allocating varargs: malloc did not align to 16 bytes.");
        }

        ObjectState *os = bindObjectInState(state, mo, true);

        for (unsigned k = funcArgs; k < callingArgs; k++) {
          if (!cb.isByValArgument(k)) {
            os->write(offsets[k], arguments[k]);
          } else {
            ConstantExpr *CE = dyn_cast<ConstantExpr>(arguments[k]);
            assert(CE); // byval argument needs to be a concrete pointer

            ObjectPair op;
            state.addressSpace.resolveOne(CE, op);
            const ObjectState *osarg = op.second;
            assert(osarg);
            for (unsigned i = 0; i < osarg->size; i++)
              os->write(offsets[k] + i, osarg->read8(i));
          }
        }
      }
    }

    unsigned numFormals = f->arg_size();
    for (unsigned k = 0; k < numFormals; k++)
      bindArgument(kf, k, state, arguments[k]);
  }
}

void Executor::transferToBasicBlock(BasicBlock *dst, BasicBlock *src, 
                                    ExecutionState &state) {
  // Note that in general phi nodes can reuse phi values from the same
  // execution of any phi nodes. this is pathological and doesn't
  // really seem to occur, but just in case we run the PhiCleanerPass
  // which makes sure this cannot happen and so it is safe to just
  // eval things in order. The PhiCleanerPass also makes sure that all
  // incoming blocks have the same order for each PHINode so we only
  // have to compute the index once.
  //
  // With that done we simply set an index in the state so that PHI
  // instructions know which argument to eval, set the pc, and continue.
  
  // XXX this lookup has to go ?
  KFunction *kf = state.stack.back().kf;
  unsigned entry = kf->basicBlockEntry[dst];
  state.pc = &kf->instructions[entry];
  if (state.pc->inst->getOpcode() == Instruction::PHI) {
    PHINode *first = static_cast<PHINode*>(state.pc->inst);
    state.incomingBBIndex = first->getBasicBlockIndex(src);
  }
}

/// Compute the true target of a function call, resolving LLVM aliases
/// and bitcasts.
Function *Executor::getTargetFunction(Value *calledVal) {
  SmallPtrSet<const GlobalValue*, 3> Visited;

  Constant *c = dyn_cast<Constant>(calledVal);
  if (!c)
    return 0;

  while (true) {
    if (GlobalValue *gv = dyn_cast<GlobalValue>(c)) {
      if (!Visited.insert(gv).second)
        return 0;

      if (Function *f = dyn_cast<Function>(gv))
        return f;
      else if (GlobalAlias *ga = dyn_cast<GlobalAlias>(gv))
        c = ga->getAliasee();
      else
        return 0;
    } else if (llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(c)) {
      if (ce->getOpcode()==Instruction::BitCast)
        c = ce->getOperand(0);
      else
        return 0;
    } else
      return 0;
  }
}

llvm::LoadInst* getLoadOperand(llvm::Value *v) {
    if (auto *li = llvm::dyn_cast<llvm::LoadInst>(v))
        return li;

    // unwrap bitcast/sext/zext/trunc
    if (auto *inst = llvm::dyn_cast<llvm::Instruction>(v)) {
        if (inst->isCast()) {
            return getLoadOperand(inst->getOperand(0));
        }
    }

    return nullptr; // not a load
}


std::pair<int, const MemoryObject *> Executor::findInductionVariable(ExecutionState &state, Instruction *ii, KLoopInfo * loopInfo) {
// for two loadInsts, iterate ir in latch, is there a store, also return the memoryobject
  int parIndex = 0;
  LoadInst *indexLoadInst = nullptr;
  LoadInst *leftLoadInst = getLoadOperand(ii->getOperand(0));
  LoadInst *rightLoadInst = getLoadOperand(ii->getOperand(1));
  bool bothSym = leftLoadInst != nullptr && rightLoadInst != nullptr;


  for (auto &ir : *loopInfo->backedge) {
    if (auto *storeInst = llvm::dyn_cast<llvm::StoreInst>(&ir)) {
      if (leftLoadInst && storeInst->getOperand(1) == leftLoadInst->getOperand(0)) {
        parIndex = 0;
        indexLoadInst = leftLoadInst;
        break;
      }
      if (rightLoadInst && storeInst->getOperand(1) == rightLoadInst->getOperand(0)) {
        parIndex = 1;
        indexLoadInst = rightLoadInst;
        break;
      }
    }
  }

  if(indexLoadInst==nullptr && !bothSym) {
    if (leftLoadInst) {
      parIndex = 0;
      indexLoadInst = leftLoadInst;
    }
    if (rightLoadInst) {
      parIndex = 1;
      indexLoadInst = rightLoadInst;
    }
  }

  if(!indexLoadInst) {
    return std::make_pair(-1, nullptr);
  }

  const MemoryObject *mo = nullptr;
  if (indexLoadInst) {
    KFunction *kf = state.stack.back().kf;
    ref<Expr> pointerExpr;

    // Start index of this block
    // Find last instruction index in this block
    unsigned endIdx;
    auto bbIter = ii->getParent()->getIterator();
    auto nextIt = std::next(bbIter);

    if (nextIt != ii->getParent()->getParent()->end()) {
      endIdx = kf->basicBlockEntry[&*nextIt] - 1;
    } else {
      endIdx = kf->numInstructions - 1; // last block in function
    }

    for (int i = endIdx; i >= 0; --i) {
      KInstruction *kbinst = kf->instructions[i];
      if (kbinst->inst == indexLoadInst) {
        pointerExpr = eval(kbinst, 0, state).value;
        break;
      }
    }

    ObjectPair op;
    bool success;
    if (pointerExpr && state.addressSpace.resolveOne(state, solver.get(), pointerExpr, op, success) && success) {
      mo = op.first;
    }
  }
  
  return std::make_pair(parIndex, mo);
}



void Executor::addBoundConstraintForArg(ExecutionState &state, KInstruction *ki) {
  Instruction *i = ki->inst;
  bool isSigned;

  if (const llvm::BinaryOperator *binOp = llvm::dyn_cast<llvm::BinaryOperator>(i)) {
    if (binOp->getOpcode() == Instruction::SDiv || binOp->getOpcode() == Instruction::SRem) {
      isSigned = true;
    } else if (binOp->getOpcode() == Instruction::UDiv || binOp->getOpcode() == Instruction::URem) {
      isSigned = false;
    } else {
      isSigned = binOp->hasNoSignedWrap();
    }
    if (isSigned) {
      int numOps = i->getNumOperands();
      if (numOps >= 1) {
        ref<Expr> left = eval(ki, 0, state).value;
        if (auto CE = dyn_cast<ConstantExpr>(left)) {
          CE->setSigned(true);
        }
      }
      if (numOps == 2) {
        ref<Expr> right = eval(ki, 1, state).value;
        if (auto CE = dyn_cast<ConstantExpr>(right)) {
          CE->setSigned(true);
        }
      }
    }
  } else if (ICmpInst *icmp = dyn_cast<ICmpInst>(i)) {
    if (icmp->isSigned()) {
      isSigned = true;
      ref<Expr> left = eval(ki, 0, state).value;
      if (auto CE = dyn_cast<ConstantExpr>(left)) {
        CE->setSigned(true);
      }
      ref<Expr> right = eval(ki, 1, state).value;
      if (auto CE = dyn_cast<ConstantExpr>(right)) {
        CE->setSigned(true);
      }
    } else if (icmp->isUnsigned()) {
      isSigned = false;
    } else {
      return;
    }
  } else
    return;

  if (state.allIntArgAddBound)
    return;

  ref<Expr> left = eval(ki, 0, state).value;
  ref<ReadExpr> leftRead = ReadExpr::extractReadExpr(left);
  if (leftRead && leftRead->updates.root && !leftRead->updates.root->name.empty()) {
    std::string argName = leftRead->updates.root->name;
    if (state.intArgAddBound.find(argName) != state.intArgAddBound.end()) {
      if (!state.intArgAddBound[argName].second) {
        ref<Expr> argExpr = state.intArgAddBound[argName].first;
        if (isSigned) {
          if (argExpr->getWidth() <= 64) {
            ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << (argExpr->getWidth()-1)) - 1, argExpr->getWidth());
            addConstraint(state, SleExpr::create(argExpr, maxValueExpr));
          }
          isIntArgSigned[argName] = true;
        } else {
          addConstraint(state, UgeExpr::create(argExpr, ConstantExpr::create(0, argExpr->getWidth())));  
          isIntArgSigned[argName] = false;
        }
        state.intArgAddBound[argName].second = true;
      }
    }
    if (!isSigned) {
      addConstraint(state, UgeExpr::create(left, ConstantExpr::create(0, left->getWidth())));  
    }
  }
  ref<Expr> right = eval(ki, 1, state).value;
  ref<ReadExpr> rightRead = ReadExpr::extractReadExpr(right);
  if (rightRead && rightRead->updates.root && !rightRead->updates.root->name.empty()) {
    std::string argName = rightRead->updates.root->name;
    if (state.intArgAddBound.find(argName) != state.intArgAddBound.end()) {
      if (!state.intArgAddBound[argName].second) {
        ref<Expr> argExpr = state.intArgAddBound[argName].first;
        if (isSigned) {
          if (argExpr->getWidth() <= 64) {
            ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << (argExpr->getWidth()-1)) - 1, argExpr->getWidth());
            addConstraint(state, SleExpr::create(argExpr, maxValueExpr));
          }
          isIntArgSigned[argName] = true;
        } else {
          addConstraint(state, UgeExpr::create(argExpr, ConstantExpr::create(0, argExpr->getWidth())));  
          isIntArgSigned[argName] = false;
        }
        state.intArgAddBound[argName].second = true;
      }
    }
    if (!isSigned) {
      addConstraint(state, UgeExpr::create(right, ConstantExpr::create(0, right->getWidth())));  
    }
  }
  for (auto item : state.intArgAddBound) {
    if (!item.second.second) {
      return;
    }
  }
  state.allIntArgAddBound = true;
}

std::string getMD(Value *V, StringRef Kind) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (auto *MD = I->getMetadata(Kind)) {
      auto *S = cast<MDString>(MD->getOperand(0));
      return S->getString().str();
    }
  }
  return "";
}

bool isBinaryOpSigned(Instruction *i){
  if (auto *BO = dyn_cast<BinaryOperator>(i)) {
    std::string opSign = getMD(BO, "signedness");
    if (opSign == "signed") {
      return true;
    }

    // check if result flows into signed variable

  }
  return false;
}

void Executor::executeInstruction(ExecutionState &state, KInstruction *ki) {
  Instruction *i = ki->inst;
  if (!isLoopCheck)
    addBoundConstraintForArg(state, ki);

  switch (i->getOpcode()) {
    // Control flow
  case Instruction::Ret: {
    ReturnInst *ri = cast<ReturnInst>(i);
    KInstIterator kcaller = state.stack.back().caller;
    Instruction *caller = kcaller ? kcaller->inst : nullptr;
    bool isVoidReturn = (ri->getNumOperands() == 0);
    ref<Expr> result = ConstantExpr::alloc(0, Expr::Bool);
    
    if (!isVoidReturn) {
      result = eval(ki, 0, state).value;
    }
    
    if (state.stack.size() <= 1) {
      assert(!caller && "caller set on initial stack frame");
      terminateStateOnExit(state);
    } else {
      state.popFrame();

      if (statsTracker)
        statsTracker->framePopped(state);

      if (InvokeInst *ii = dyn_cast<InvokeInst>(caller)) {
        transferToBasicBlock(ii->getNormalDest(), caller->getParent(), state);
      } else {
        state.pc = kcaller;
        ++state.pc;
      }

#ifdef SUPPORT_KLEE_EH_CXX
      if (ri->getFunction()->getName() == "_klee_eh_cxx_personality") {
        assert(dyn_cast<ConstantExpr>(result) &&
               "result from personality fn must be a concrete value");

        auto *sui = dyn_cast_or_null<SearchPhaseUnwindingInformation>(
            state.unwindingInformation.get());
        assert(sui && "return from personality function outside of "
                      "search phase unwinding");

        // unbind the MO we used to pass the serialized landingpad
        state.addressSpace.unbindObject(sui->serializedLandingpad);
        sui->serializedLandingpad = nullptr;

        if (result->isZero()) {
          // this lpi doesn't handle the exception, continue the search
          unwindToNextLandingpad(state);
        } else {
          // remember the stack index and switch to clean-up phase
          state.unwindingInformation =
              std::make_unique<CleanupPhaseUnwindingInformation>(
                  sui->exceptionObject, cast<ConstantExpr>(result),
                  sui->unwindingProgress);
          // this pointer is now invalidated
          sui = nullptr;
          unwindToNextLandingpad(state);
        }

        // never return normally from the personality fn
        break;
      }
#endif // SUPPORT_KLEE_EH_CXX

      if (!isVoidReturn) {
        Type *t = caller->getType();
        if (t != Type::getVoidTy(i->getContext())) {
          // may need to do coercion due to bitcasts
          Expr::Width from = result->getWidth();
          Expr::Width to = getWidthForLLVMType(t);
            
          if (from != to) {
            const CallBase &cb = cast<CallBase>(*caller);

            // XXX need to check other param attrs ?
            bool isSExt = cb.hasRetAttr(llvm::Attribute::SExt);
            if (isSExt) {
              result = SExtExpr::create(result, to);
            } else {
              result = ZExtExpr::create(result, to);
            }
          }

          bindLocal(kcaller, state, result);
        }
      } else {
        // We check that the return value has no users instead of
        // checking the type, since C defaults to returning int for
        // undeclared functions.
        if (!caller->use_empty()) {
          terminateStateOnExecError(state, "return void when caller expected a result");
        }
      }
    } 
    if (isVoidReturn) {
      std::vector<ref<Expr>> noArgs;
      specialFunctionHandler->handleCloseMerge(state, ki, noArgs);
    }     
    break;
  }
  case Instruction::Br: {
    BranchInst *bi = cast<BranchInst>(i);
    if (bi->isUnconditional()) {
      transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), state);
    } else {
      // FIXME: Find a way that we don't have this hidden dependency.
      assert(bi->getCondition() == bi->getOperand(0) &&
             "Wrong operand index!");
      
      
      ref<Expr> cond = eval(ki, 0, state).value;
      cond = optimizer.optimizeExpr(cond, false);

      BasicBlock *currentBlock = bi->getParent();
      auto it = state.loopInfoMap.find(currentBlock);
      bool isSymLoop = false;
      if (it!=state.loopInfoMap.end() && it->second != nullptr && state.loopInfoMap[currentBlock]->symbolizeIndex==true && !state.loopInfoMap[currentBlock]->indexExpr.isNull()){
        isSymLoop = true;
        if (state.loopInfoMap[currentBlock]->times>0) {
          state.loopInfoMap[currentBlock]->times = 0;
          KLoopInfo *loopInfo = state.loopInfoMap[currentBlock];
          checkDataRace(state, loopInfo);
          if (Expr::isRelatedWithLoopIndex(loopInfo->cond, loopInfo->indexExpr)) {
            ObjectPair op;
            bool success;
            if (!state.addressSpace.resolveOne(state, solver.get(), loopInfo->address, op, success) || !success) {
              terminateStateOnProgramError(state, "not found loop index object", StateTerminationType::ReportError);
              return;
            }

            ref<Expr> newIndexExpr = symbolizeIndex(state, loopInfo->indexName+"_out", op.first);
            std::map< ref<Expr>, ref<Expr> > replacements;
            replacements[loopInfo->indexExpr] = SExtExpr::create(newIndexExpr, loopInfo->indexExpr->getWidth());
            ref<Expr> newCon = ConstraintManager::replaceReadExpr(loopInfo->cond, replacements);
            addConstraint(state, Expr::createIsZero(newCon));
          }
          state.loopInfoMap[currentBlock]->indexExpr = nullptr;
          transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), state);
          break;
        }
        state.loopInfoMap[currentBlock]->times+=1;
        state.loopInfoMap[currentBlock]->cond = cond;
      } 
      else if (state.loopInfoMap.find(currentBlock)!=state.loopInfoMap.end() && it->second != nullptr){
        if (state.loopInfoMap[currentBlock]->executedTimes>=loopSymMax){
          state.loopInfoMap[currentBlock]->executedTimes = 0;
          KLoopInfo *loopInfo = state.loopInfoMap[currentBlock];
          checkDataRace(state, loopInfo);
          transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), state);
          break;
        }
      }

      Executor::StatePair branches = fork(state, cond, false, BranchType::Conditional);

      if (state.loopInfoMap.find(currentBlock)!=state.loopInfoMap.end() && (!state.loopInfoMap[currentBlock]->symbolizeIndex || !state.loopInfoMap[currentBlock]->indexExpr)){
          state.loopInfoMap[currentBlock]->executedTimes+=1;
      }

      if (it!=state.loopInfoMap.end() && it->second != nullptr && state.loopInfoMap[currentBlock]->indexExpr.isNull() && branches.second) {
        checkDataRace(*branches.second, state.loopInfoMap[currentBlock]);
        branches.second->loopInfoMap[currentBlock]->times = 0;
        branches.second->loopInfoMap[currentBlock]->indexExpr = nullptr;
      }

      if (isSymLoop) {
        KLoopInfo *loopInfo = state.loopInfoMap[currentBlock];
        BasicBlock *backedge = loopInfo->backedge;
        KFunction *kf = state.stack.back().kf;

        if (!loopInfo->twoBlocks) {
        for (unsigned i = kf->basicBlockEntry[backedge]; i < kf->numInstructions; ++i) {
          KInstruction *kbinst = kf->instructions[i];
          if (kbinst->inst->getParent() == backedge) {
            if (kbinst->inst->getOpcode() == Instruction::Store) {
              ref<Expr> address = eval(kbinst, 1, state).value;
              if (address != loopInfo->address) {
                isLoopCheck = true;
                executeInstruction(state, kbinst);
                isLoopCheck = false;
                continue;
              }
              
              ref<Expr> value = eval(kbinst, 0, state).value;
              ref<Expr> intialVal = loopInfo->initialVal;
              ref<Expr> indexExpr = loopInfo->indexExpr;
              if (isa<ExtractExpr>(value) || isa<ZExtExpr>(value) || isa<SExtExpr>(value)) {
                value = value->getKid(0);
              }
              if (AddExpr *addExpr = dyn_cast<AddExpr>(value)) {
                ref<Expr> op1 = addExpr->getKid(0);
                ref<Expr> op2 = addExpr->getKid(1);
                ref<Expr> increment;
                if (op1 == indexExpr) {
                  increment = op2;
                } else {
                  increment = op1;
                }
                loopInfo->increment = increment;
                loopInfo->increType = KLoopInfo::IncreType::ADD;
                
                ref<Expr> indexConstraint = EqExpr::create(SRemExpr::create(SExtExpr::create(SubExpr::create(SExtExpr::create(indexExpr, intialVal->getWidth()), intialVal), increment->getWidth()), increment), ConstantExpr::create(0, increment->getWidth()));
                addConstraint(state, indexConstraint);
              } else if (SubExpr *subExpr = dyn_cast<SubExpr>(value)) {
                loopInfo->increType = KLoopInfo::IncreType::SUB;
                loopInfo->increment = subExpr->getKid(1);
                ref<Expr> indexConstraint = EqExpr::create(SRemExpr::create(SExtExpr::create(SubExpr::create(intialVal, SExtExpr::create(indexExpr, intialVal->getWidth())), loopInfo->increment->getWidth()), subExpr->getKid(1)), ConstantExpr::create(0, loopInfo->increment->getWidth()));
                addConstraint(state, indexConstraint);
              } 

              break;
            } else if (kbinst->inst->getOpcode() != Instruction::Br){
              isLoopCheck = true;
              executeInstruction(state, kbinst);
              isLoopCheck = false;
            }
          } else {
            break;
          }
        }
      } else {
        for (unsigned i = kf->basicBlockEntry[backedge]; i < kf->numInstructions; ++i) {
          KInstruction *kbinst = kf->instructions[i];
          if (kbinst->inst->getParent() == backedge) {
            if (kbinst->inst->getOpcode() == Instruction::Load) {
              ref<Expr> address = eval(kbinst, 0, state).value;
              if (!address.isNull() && address == loopInfo->address) {
                bindLocal(kbinst, state, loopInfo->indexExpr);
              }
            } else if (kbinst->inst->getOpcode() == Instruction::Store) {
              ref<Expr> address = eval(kbinst, 1, state).value;
              if (address.isNull() || address != loopInfo->address) {
                continue;
              }
              
              ref<Expr> value = eval(kbinst, 0, state).value;
              ref<Expr> intialVal = loopInfo->initialVal;
              ref<Expr> indexExpr = loopInfo->indexExpr;
 
              if (isa<ExtractExpr>(value) || isa<ZExtExpr>(value) || isa<SExtExpr>(value)) {
                value = value->getKid(0);
              }
              if (AddExpr *addExpr = dyn_cast<AddExpr>(value)) {
                ref<Expr> op1 = addExpr->getKid(0);
                ref<Expr> op2 = addExpr->getKid(1);
                ref<Expr> increment;
                if (op1 == indexExpr) {
                  increment = op2;
                } else {
                  increment = op1;
                }
                loopInfo->increment = increment;
                loopInfo->increType = KLoopInfo::IncreType::ADD;
                
                ref<Expr> indexConstraint = EqExpr::create(SRemExpr::create(SExtExpr::create(SubExpr::create(SExtExpr::create(indexExpr, intialVal->getWidth()), intialVal), increment->getWidth()), increment), ConstantExpr::create(0, increment->getWidth()));
                addConstraint(state, indexConstraint);
              } else if (SubExpr *subExpr = dyn_cast<SubExpr>(value)) {
                loopInfo->increType = KLoopInfo::IncreType::SUB;
                loopInfo->increment = subExpr->getKid(1);
                ref<Expr> indexConstraint = EqExpr::create(SRemExpr::create(SExtExpr::create(SubExpr::create(intialVal, SExtExpr::create(indexExpr, intialVal->getWidth())), loopInfo->increment->getWidth()), subExpr->getKid(1)), ConstantExpr::create(0, loopInfo->increment->getWidth()));
                addConstraint(state, indexConstraint);
              }
            } else if (kbinst->inst->getOpcode() == Instruction::Call) {
              const CallBase &cb = cast<CallBase>(*(kbinst->inst));
              unsigned numArgs = cb.arg_size();
              if (numArgs == 0) {
                isLoopCheck = true;
                executeInstruction(state, kbinst);
                isLoopCheck = false;
              }
            } else if (kbinst->inst->getOpcode() != Instruction::Br){
              unsigned numOperands = kbinst->inst->getNumOperands();
              bool isValNull = false;
              for (unsigned i = 0; i < numOperands; ++i) {
                ref<Expr> value = eval(kbinst, i, state).value;
                if (value.isNull()) {
                  isValNull = true;
                  break;
                }
              }
              if (!isValNull) {
                isLoopCheck = true;
                executeInstruction(state, kbinst);
                isLoopCheck = false;
              }
            }
          } else {
            break;
          }
        }
      }
      }

      // NOTE: There is a hidden dependency here, markBranchVisited
      // requires that we still be in the context of the branch
      // up with convenient instruction specific data.
      if (statsTracker && state.stack.back().kf->trackCoverage)
        statsTracker->markBranchVisited(branches.first, branches.second);

      if (!isSymLoop && !pred_empty(currentBlock) && branches.second) {
        for (auto PI = pred_begin(currentBlock), PE = pred_end(currentBlock); PI != PE; ++PI) {
          BasicBlock *pred = *PI;
          if (state.loopInfoMap.find(pred)!=state.loopInfoMap.end()){
            KLoopInfo *loopInfo = state.loopInfoMap[pred];
            if (loopInfo->symbolizeIndex==true && !loopInfo->indexExpr.isNull() && loopInfo->bodyBlocks.find(bi->getSuccessor(1)) == loopInfo->bodyBlocks.end()) {
                isSymLoop = true;
                break;
            }
          }
        }
      }
      if (branches.first)
        transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
      if (branches.second) {
        if (isSymLoop && branches.first) {
          terminateStateEarlyUser(*branches.second, "symbolic loop first out");
        } else
          transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
      }
    }
    break;
  }
  case Instruction::IndirectBr: {
    // implements indirect branch to a label within the current function
    const auto bi = cast<IndirectBrInst>(i);
    auto address = eval(ki, 0, state).value;
    address = toUnique(state, address);

    // concrete address
    if (const auto CE = dyn_cast<ConstantExpr>(address.get())) {
      const auto bb_address = (BasicBlock *) CE->getZExtValue(Context::get().getPointerWidth());
      transferToBasicBlock(bb_address, bi->getParent(), state);
      break;
    }

    // symbolic address
    const auto numDestinations = bi->getNumDestinations();
    std::vector<BasicBlock *> targets;
    targets.reserve(numDestinations);
    std::vector<ref<Expr>> expressions;
    expressions.reserve(numDestinations);

    ref<Expr> errorCase = ConstantExpr::alloc(1, Expr::Bool);
    SmallPtrSet<BasicBlock *, 5> destinations;
    // collect and check destinations from label list
    for (unsigned k = 0; k < numDestinations; ++k) {
      // filter duplicates
      const auto d = bi->getDestination(k);
      if (destinations.count(d)) continue;
      destinations.insert(d);

      // create address expression
      const auto PE = Expr::createPointer(reinterpret_cast<std::uint64_t>(d));
      ref<Expr> e = EqExpr::create(address, PE);

      // exclude address from errorCase
      errorCase = AndExpr::create(errorCase, Expr::createIsZero(e));

      // check feasibility
      bool result;
      bool success __attribute__((unused)) =
          solver->mayBeTrue(state.constraints, e, result, state.queryMetaData, state.intArrNames, state.getIOExprs());
      assert(success && "FIXME: Unhandled solver failure");
      if (result) {
        targets.push_back(d);
        expressions.push_back(e);
      }
    }
    // check errorCase feasibility
    bool result;
    bool success __attribute__((unused)) = solver->mayBeTrue(
        state.constraints, errorCase, result, state.queryMetaData, state.intArrNames, state.getIOExprs());
    assert(success && "FIXME: Unhandled solver failure");
    if (result) {
      expressions.push_back(errorCase);
    }

    // fork states
    std::vector<ExecutionState *> branches;
    branch(state, expressions, branches, BranchType::Indirect);

    // terminate error state
    if (result) {
      terminateStateOnExecError(*branches.back(), "indirectbr: illegal label address");
      branches.pop_back();
    }

    // branch states to resp. target blocks
    assert(targets.size() == branches.size());
    for (std::vector<ExecutionState *>::size_type k = 0; k < branches.size(); ++k) {
      if (branches[k]) {
        transferToBasicBlock(targets[k], bi->getParent(), *branches[k]);
      }
    }

    break;
  }
  case Instruction::Switch: {
    SwitchInst *si = cast<SwitchInst>(i);
    ref<Expr> cond = eval(ki, 0, state).value;
    BasicBlock *bb = si->getParent();

    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(cond)) {
      // Somewhat gross to create these all the time, but fine till we
      // switch to an internal rep.
      llvm::IntegerType *Ty = cast<IntegerType>(si->getCondition()->getType());
      ConstantInt *ci = ConstantInt::get(Ty, CE->getZExtValue());
      unsigned index = si->findCaseValue(ci)->getSuccessorIndex();
      transferToBasicBlock(si->getSuccessor(index), si->getParent(), state);
    } else {
      // Handle possible different branch targets

      // We have the following assumptions:
      // - each case value is mutual exclusive to all other values
      // - order of case branches is based on the order of the expressions of
      //   the case values, still default is handled last
      std::vector<BasicBlock *> bbOrder;
      std::map<BasicBlock *, ref<Expr> > branchTargets;

      std::map<ref<Expr>, BasicBlock *> expressionOrder;

      // Iterate through all non-default cases and order them by expressions
      for (auto i : si->cases()) {
        ref<Expr> value = evalConstant(i.getCaseValue());

        BasicBlock *caseSuccessor = i.getCaseSuccessor();
        expressionOrder.insert(std::make_pair(value, caseSuccessor));
      }

      // Track default branch values
      ref<Expr> defaultValue = ConstantExpr::alloc(1, Expr::Bool);


      // iterate through all non-default cases but in order of the expressions
      for (std::map<ref<Expr>, BasicBlock *>::iterator
               it = expressionOrder.begin(),
               itE = expressionOrder.end();
           it != itE; ++it) {
        ref<Expr> match = EqExpr::create(cond, it->first);

        // skip if case has same successor basic block as default case
        if (it->second == si->getDefaultDest()) continue;

        // Make sure that the default value does not contain this target's value
        defaultValue = AndExpr::create(defaultValue, Expr::createIsZero(match));

        // Check if control flow could take this case
        bool result;
        match = optimizer.optimizeExpr(match, false);
        bool success = solver->mayBeTrue(state.constraints, match, result,
                                         state.queryMetaData, state.intArrNames, state.getIOExprs());
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (result) {

          BasicBlock *caseSuccessor = it->second;

          // Handle the case that a basic block might be the target of multiple
          // switch cases.
          // Currently we generate an expression containing all switch-case
          // values for the same target basic block. We spare us forking too
          // many times but we generate more complex condition expressions
          // TODO Add option to allow to choose between those behaviors
          std::pair<std::map<BasicBlock *, ref<Expr> >::iterator, bool> res =
              branchTargets.insert(std::make_pair(
                  caseSuccessor, ConstantExpr::alloc(0, Expr::Bool)));

          res.first->second = OrExpr::create(match, res.first->second);

          // Only add basic blocks which have not been target of a branch yet
          if (res.second) {
            bbOrder.push_back(caseSuccessor);
          }
        }
      }

      // Check if control could take the default case
      defaultValue = optimizer.optimizeExpr(defaultValue, false);
      bool res;
      bool success = solver->mayBeTrue(state.constraints, defaultValue, res,
                                       state.queryMetaData, state.intArrNames, state.getIOExprs());
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        std::pair<std::map<BasicBlock *, ref<Expr> >::iterator, bool> ret =
            branchTargets.insert(
                std::make_pair(si->getDefaultDest(), defaultValue));
        if (ret.second) {
          bbOrder.push_back(si->getDefaultDest());
        }
      }

      // Fork the current state with each state having one of the possible
      // successors of this switch
      std::vector< ref<Expr> > conditions;
      for (std::vector<BasicBlock *>::iterator it = bbOrder.begin(),
                                               ie = bbOrder.end();
           it != ie; ++it) {
        conditions.push_back(branchTargets[*it]);
      }
      std::vector<ExecutionState*> branches;
      branch(state, conditions, branches, BranchType::Switch);

      std::vector<ExecutionState*>::iterator bit = branches.begin();
      for (std::vector<BasicBlock *>::iterator it = bbOrder.begin(),
                                               ie = bbOrder.end();
           it != ie; ++it) {
        ExecutionState *es = *bit;
        if (es)
          transferToBasicBlock(*it, bb, *es);
        ++bit;
      }
    }
    break;
  }
  case Instruction::Unreachable:
    // Note that this is not necessarily an internal bug, llvm will
    // generate unreachable instructions in cases where it knows the
    // program will crash. So it is effectively a SEGV or internal
    // error.
    terminateStateOnExecError(state, "reached \"unreachable\" instruction");
    break;

  case Instruction::Invoke:
  case Instruction::Call: {
    // Ignore debug intrinsic calls
    if (isa<DbgInfoIntrinsic>(i))
      break;
    
    const CallBase &cb = cast<CallBase>(*i);
    Value *fp = cb.getCalledOperand();
    unsigned numArgs = cb.arg_size();
    Function *f = getTargetFunction(fp);


    // evaluate arguments
    std::vector< ref<Expr> > arguments;
    arguments.reserve(numArgs);

    for (unsigned j=0; j<numArgs; ++j)
      arguments.push_back(eval(ki, j+1, state).value);

    if (auto* asmValue = dyn_cast<InlineAsm>(fp)) { //TODO: move to `executeCall`
      if (ExternalCalls != ExternalCallPolicy::None) {
        KInlineAsm callable(asmValue);
        callExternalFunction(state, ki, &callable, arguments);
      } else {
        terminateStateOnExecError(state, "external calls disallowed (in particular inline asm)");
      }
      break;
    }

    if (f) {
      const FunctionType *fType = f->getFunctionType();
#if LLVM_VERSION_MAJOR >= 15
      const FunctionType *fpType = cb.getFunctionType();
#else
      const FunctionType *fpType =
          dyn_cast<FunctionType>(fp->getType()->getPointerElementType());
#endif

      // special case the call with a bitcast case
      if (fType != fpType) {
        assert(fType && fpType && "unable to get function type");

        // XXX check result coercion

        // XXX this really needs thought and validation
        unsigned i=0;
        for (std::vector< ref<Expr> >::iterator
               ai = arguments.begin(), ie = arguments.end();
             ai != ie; ++ai) {
          Expr::Width to, from = (*ai)->getWidth();
            
          if (i<fType->getNumParams()) {
            to = getWidthForLLVMType(fType->getParamType(i));

            if (from != to) {
              // XXX need to check other param attrs ?
              bool isSExt = cb.paramHasAttr(i, llvm::Attribute::SExt);
              if (isSExt) {
                arguments[i] = SExtExpr::create(arguments[i], to);
              } else {
                arguments[i] = ZExtExpr::create(arguments[i], to);
              }
            }
          }
            
          i++;
        }
      }

      executeCall(state, ki, f, arguments);
    } else {
      ref<Expr> v = eval(ki, 0, state).value;

      ExecutionState *free = &state;
      bool hasInvalid = false, first = true;

      /* XXX This is wasteful, no need to do a full evaluate since we
         have already got a value. But in the end the caches should
         handle it for us, albeit with some overhead. */
      do {
        v = optimizer.optimizeExpr(v, true);
        ref<ConstantExpr> value;
        bool success =
            solver->getValue(free->constraints, v, value, free->queryMetaData);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        StatePair res = fork(*free, EqExpr::create(v, value), true, BranchType::Call);
        if (res.first) {
          uint64_t addr = value->getZExtValue();
          auto it = legalFunctions.find(addr);
          if (it != legalFunctions.end()) {
            f = it->second;

            // Don't give warning on unique resolution
            if (res.second || !first)
              klee_warning_once(reinterpret_cast<void*>(addr),
                                "resolved symbolic function pointer to: %s",
                                f->getName().data());

            executeCall(*res.first, ki, f, arguments);
          } else {
            if (!hasInvalid) {
              terminateStateOnExecError(state, "invalid function pointer");
              hasInvalid = true;
            }
          }
        }

        first = false;
        free = res.second;
      } while (free);
    }
    break;
  }
  case Instruction::PHI: {
    ref<Expr> result = eval(ki, state.incomingBBIndex, state).value;
    bindLocal(ki, state, result);
    break;
  }

    // Special instructions
  case Instruction::Select: {
    // NOTE: It is not required that operands 1 and 2 be of scalar type.
    ref<Expr> cond = eval(ki, 0, state).value;
    ref<Expr> tExpr = eval(ki, 1, state).value;
    ref<Expr> fExpr = eval(ki, 2, state).value;
    ref<Expr> result = SelectExpr::create(cond, tExpr, fExpr);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::VAArg:
    terminateStateOnExecError(state, "unexpected VAArg instruction");
    break;

    // Arithmetic / logical

  case Instruction::Add: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AddExpr::create(left, right);

    std::string exprStr;
    llvm::raw_string_ostream os(exprStr);
    os << result;
    os.flush(); // optional, os.str() also works
    bool hasTemp = exprStr.find("temp_storage") != std::string::npos;
    
    if (isLoopCheck || hasTemp) {
      bindLocal(ki, state, result);
      break;
    }

    std::string arrayName = "";
    const Array *array0 = nullptr;
    ref<Expr> index0 = ConcatExpr::getArrayIndex(left, &array0);
    const Array *array1 = nullptr;
    ref<Expr> index1 = ConcatExpr::getArrayIndex(right, &array1);
    ref<Expr> add_expr = nullptr;
    if (!index0.isNull()) {
      arrayName = array0->name;
      add_expr = right;
    } else if (!index1.isNull()) {
      arrayName = array1->name;
      add_expr = left;
    }

    if (!arrayName.empty()) {
      if (state.symbolicArrayMap.find(arrayName) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[arrayName];
        if (symmo->maxVal.isNull()) {
          if (auto add_expr_con = dyn_cast<ConstantExpr>(add_expr)) {
            if (add_expr_con->getZExtValue() <= 1) {
              bindLocal(ki, state, result);
              break;
            }
          }
          // bindLocal(ki, state, result);
          // break;
        }
      }
      if (arrayName.find("temp_storage") != std::string::npos) {
        bindLocal(ki, state, result);
        break;
      }
    }

    if (const llvm::BinaryOperator *binOp = llvm::dyn_cast<llvm::BinaryOperator>(i)) {
      bool isSigned = binOp->hasNoSignedWrap() || isBinaryOpSigned(i);
      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        if (isSigned) {
          leftCE->setSigned(true);
        }
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        if (isSigned) {
          rightCE->setSigned(true);
        }
      }
      result = AddExpr::create(left, right, isSigned);

      llvm::Type *resultType = binOp->getType();
      if (resultType->isIntegerTy()) {
        unsigned bitWidth = resultType->getIntegerBitWidth();
        if (bitWidth <= 32) {           
          
          ref<Expr> overflowCheck;
          ref<Expr> maxValueExpr;
          ref<Expr> realRes;
          if (isSigned) {
            uint64_t maxValue = (1ULL << (bitWidth - 1)) - 1; 
            maxValueExpr = ConstantExpr::create(maxValue, Expr::Int64);
            realRes = AddExpr::create(SExtExpr::create(left, Expr::Int64), SExtExpr::create(right, Expr::Int64));
            overflowCheck = SleExpr::create(realRes, maxValueExpr);
          } else {
            maxValueExpr = ConstantExpr::create((1ULL << bitWidth) - 1, Expr::Int64);
            realRes = AddExpr::create(ZExtExpr::create(left, Expr::Int64), ZExtExpr::create(right, Expr::Int64));
            overflowCheck = UleExpr::create(realRes, maxValueExpr);
          }

          StatePair branches = fork(state, overflowCheck, true, BranchType::Trunc);

          ExecutionState *bound = branches.first;
          if (bound) {
            bindLocal(ki, state, result);
          }

          ExecutionState *unbound = branches.second;
          if(unbound) {
            terminateStateOnProgramError(*unbound, "integer overflow", StateTerminationType::Overflow);
          }
          break;
        }
      }
    }
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Sub: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = SubExpr::create(left, right);

    if (const llvm::BinaryOperator *binOp = llvm::dyn_cast<llvm::BinaryOperator>(i)) {
      bool isSigned = !binOp->hasNoUnsignedWrap();
      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        if (isSigned) {
          leftCE->setSigned(true);
        }
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        if (isSigned) {
          rightCE->setSigned(true);
        }
      }
      result = SubExpr::create(left, right, isSigned);
    }
    bindLocal(ki, state, result);
    break;
  }
 
  case Instruction::Mul: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = MulExpr::create(left, right);
    
    if (isLoopCheck) {
      bindLocal(ki, state, result);
      break;
    }

    // std::string arrayName = "";
    // const Array *array0 = nullptr;
    // ref<Expr> index0 = ConcatExpr::getArrayIndex(left, &array0);
    // const Array *array1 = nullptr;
    // ref<Expr> index1 = ConcatExpr::getArrayIndex(right, &array1);
    // if (!index0.isNull()) {
    //   arrayName = array0->name;
    // } else if (!index1.isNull()) {
    //   arrayName = array1->name;
    // }

    // if (!arrayName.empty()) {
    //   if (state.symbolicArrayMap.find(arrayName) != state.symbolicArrayMap.end()) {
    //     SymArrayMemoryObject *symmo = state.symbolicArrayMap[arrayName];
    //     if (symmo->maxVal.isNull()) {
    //       bindLocal(ki, state, result);
    //       break;
    //     }
    //   }
    // }

    if (const llvm::BinaryOperator *binOp = llvm::dyn_cast<llvm::BinaryOperator>(i)) {
      bool isSigned = binOp->hasNoSignedWrap() || isBinaryOpSigned(i);
      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        if (isSigned) {
          leftCE->setSigned(true);
        }
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        if (isSigned) {
          rightCE->setSigned(true);
        }
      }
      result = MulExpr::create(left, right, isSigned);

      llvm::Type *resultType = binOp->getType();
      if (resultType->isIntegerTy()) {
        unsigned bitWidth = resultType->getIntegerBitWidth();
        if (bitWidth <= 32) { 
          
          ref<Expr> overflowCheck;
          ref<Expr> maxValueExpr;
          ref<Expr> realRes;
          if (isSigned) {
            uint64_t maxValue = (1ULL << (bitWidth - 1)) - 1; 
            maxValueExpr = ConstantExpr::create(maxValue, Expr::Int64);
            realRes = MulExpr::create(SExtExpr::create(left, Expr::Int64), SExtExpr::create(right, Expr::Int64));
            overflowCheck = SleExpr::create(realRes, maxValueExpr);
          } else {
            maxValueExpr = ConstantExpr::create((1ULL << bitWidth) - 1, Expr::Int64);
            realRes = MulExpr::create(ZExtExpr::create(left, Expr::Int64), ZExtExpr::create(right, Expr::Int64));
            overflowCheck = UleExpr::create(realRes, maxValueExpr);
          }

          StatePair branches = fork(state, overflowCheck, true, BranchType::Trunc);
          ExecutionState *bound = branches.first;
          if (bound) {
            bindLocal(ki, state, result);
          }

          ExecutionState *unbound = branches.second;
          if(unbound) {
            terminateStateOnProgramError(*unbound, "integer overflow", StateTerminationType::Overflow);
          }
          break;
        }
      }
    }
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::UDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = UDivExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::SDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
      leftCE->setSigned(true);
    }
    if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
      rightCE->setSigned(true);
    }
    ref<Expr> result = SDivExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::URem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = URemExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::SRem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
      leftCE->setSigned(true);
    }
    if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
      rightCE->setSigned(true);
    }
    ref<Expr> result = SRemExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::And: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AndExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Or: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = OrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Xor: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = XorExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Shl: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = ShlExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::LShr: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = LShrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::AShr: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AShrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

    // Compare

  case Instruction::ICmp: {
    CmpInst *ci = cast<CmpInst>(i);
    ICmpInst *ii = cast<ICmpInst>(ci);


    BasicBlock *currentBlock = ci->getParent();
    bool needSymbolic = false;
    KLoopInfo * loopInfo;
    if (state.loopInfoMap.find(currentBlock)!=state.loopInfoMap.end() && state.loopInfoMap[currentBlock] != nullptr && state.loopInfoMap[currentBlock]->symbolizeIndex==true && state.loopInfoMap[currentBlock]->times==0){
      needSymbolic = true;
      loopInfo = state.loopInfoMap[currentBlock];
      if (loopInfo->indexExpr || loopInfo->address) {
        loopInfo->incrIndexName();
      }
      if (state.kernelConfig.getThreadIdx().x &&
          state.addressInLoop.find(loopInfo->indexName) == state.addressInLoop.end()) {
        std::map<ref<Expr>, std::tuple<ExecutionState::MemOp, MemoryObject::MemType, std::string>> addressMap;
        state.addressInLoop[loopInfo->indexName] = addressMap;
      }
    }

    switch(ii->getPredicate()) {
    case ICmpInst::ICMP_EQ: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;



      bool isLarge = true;
      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right) && needSymbolic) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int64_t leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int64_t rightV = rightCE->getZExtValue();
        if (leftV < rightV)
          isLarge = false;

        if (leftV > (1ULL << 31) - 1) {
          bool success;
          ObjectPair op;
          if (state.addressSpace.resolveOne(state, solver.get(), leftCE, op, success) && success) {
            if (state.loopInfoMap.find(currentBlock) != state.loopInfoMap.end()) {
              state.loopInfoMap.erase(currentBlock);
              ref<Expr> result = EqExpr::create(left, right);
              bindLocal(ki, state, result);
              break;
            }
          }
        }

        if (rightV > (1ULL << 31) - 1) {
          bool success;
          ObjectPair op;
          if (state.addressSpace.resolveOne(state, solver.get(), rightCE, op, success) && success) {
            if (state.loopInfoMap.find(currentBlock) != state.loopInfoMap.end()) {
              state.loopInfoMap.erase(currentBlock);
              ref<Expr> result = EqExpr::create(left, right);
              bindLocal(ki, state, result);
              break;
            }
          }
        }

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = EqExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
          } else {
            loopInfo->initialVal = right;
          }
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo, true, loopInfo->initialVal);
          indexExpr = SExtExpr::create(indexExpr, right->getWidth());
          ref<Expr> result;
          if (inductionRes.first==0){
            loopInfo->exitExpr = right;
            result = EqExpr::create(indexExpr, right);
            if (isLarge) {
              addConstraint(state, UleExpr::create(indexExpr, left));
              addConstraint(state, UgeExpr::create(indexExpr, right));
            } else {
              addConstraint(state, UgeExpr::create(indexExpr, left));
              addConstraint(state, UleExpr::create(indexExpr, right));
            }
          } else {
            loopInfo->exitExpr = left;
            result = EqExpr::create(indexExpr, left);
            if (isLarge) {
              addConstraint(state, UgeExpr::create(indexExpr, right));
              addConstraint(state, UleExpr::create(indexExpr, left));
            } else {
              addConstraint(state, UgeExpr::create(indexExpr, left));
              addConstraint(state, UleExpr::create(indexExpr, right));
            }
          }
          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }

      ref<Expr> result = EqExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_NE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;

      bool isLarge = true;
      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right) && needSymbolic) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();
        if (leftV < rightV)
          isLarge = false;

        if (std::abs(rightV-leftV) <= loopSymMax || (rightV && leftV)) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = NeExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
          } else {
            loopInfo->initialVal = right;
          }
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo, true, loopInfo->initialVal);
          indexExpr = SExtExpr::create(indexExpr, right->getWidth());
          ref<Expr> result;
          if (inductionRes.first==0){
            loopInfo->exitExpr = right;
            result = NeExpr::create(indexExpr, right);
            if (isLarge) {
              addConstraint(state, UleExpr::create(indexExpr, left));
              addConstraint(state, UgeExpr::create(indexExpr, right));
            } else {
              addConstraint(state, UgeExpr::create(indexExpr, left));
              addConstraint(state, UleExpr::create(indexExpr, right));
            }
          } else {
            loopInfo->exitExpr = left;
            result = NeExpr::create(indexExpr, left);
            if (isLarge) {
              addConstraint(state, UleExpr::create(indexExpr, left));
              addConstraint(state, UgeExpr::create(indexExpr, right));
            } else {
              addConstraint(state, UgeExpr::create(indexExpr, left));
              addConstraint(state, UleExpr::create(indexExpr, right));
            }
          }
          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }        
      }
      ref<Expr> result = NeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_UGT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;

      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right) && needSymbolic) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = UgtExpr::create(left, right);
          bindLocal(ki, state,result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo);
          indexExpr = ZExtExpr::create(indexExpr, right->getWidth());
          ref<Expr> result;
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = false;
            result = UgtExpr::create(indexExpr, right);
            addConstraint(state, UleExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = false;
            result = UgtExpr::create(left, indexExpr);
            addConstraint(state, UgeExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state,result);
          break;
        }
      }

      ref<Expr> result = UgtExpr::create(left, right);
      bindLocal(ki, state,result);
      break;
    }

    case ICmpInst::ICMP_UGE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;

      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = UgeExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo);
          indexExpr = ZExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = false;
            result = UgeExpr::create(indexExpr, right);
            addConstraint(state, UleExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = false;
            result = UgeExpr::create(left, indexExpr);
            addConstraint(state, UgeExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      ref<Expr> result = UgeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;


      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = UltExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo);
          indexExpr = ZExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = false;
            result = UltExpr::create(indexExpr, right);
            addConstraint(state, UgeExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = false;
            result = UltExpr::create(left, indexExpr);
            addConstraint(state, UleExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      ref<Expr> result = UltExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;

      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = UleExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo);
          indexExpr = ZExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = false;
            result = UleExpr::create(indexExpr, right);
            addConstraint(state, UgeExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = false;
            result = UleExpr::create(left, indexExpr);
            addConstraint(state, UleExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      ref<Expr> result = UleExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        leftCE->setSigned(true);
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        rightCE->setSigned(true);
      }


      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = SgtExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo, true);
          indexExpr = SExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = true;
            result = SgtExpr::create(indexExpr, right);
            addConstraint(state, SleExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = true;
            result = SgtExpr::create(left, indexExpr);
            addConstraint(state, SgeExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      ref<Expr> result = SgtExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        leftCE->setSigned(true);
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        rightCE->setSigned(true);
      }

      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = SgeExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }

      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo, true);
          indexExpr = SExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = true;
            result = SgeExpr::create(indexExpr, right);
            addConstraint(state, SleExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = true;
            result = SgeExpr::create(left, indexExpr);
            addConstraint(state, SgeExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      ref<Expr> result = SgeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;

      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        leftCE->setSigned(true);
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        rightCE->setSigned(true);
      }

      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = SltExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }
      
      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo, true);
          indexExpr = SExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = true;
            result = SltExpr::create(indexExpr, right);
            addConstraint(state, SgeExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = true;
            result = SltExpr::create(left, indexExpr);
            addConstraint(state, SleExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      ref<Expr> result = SltExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      if (auto leftCE = dyn_cast<ConstantExpr>(left)) {
        leftCE->setSigned(true);
      }
      if (auto rightCE = dyn_cast<ConstantExpr>(right)) {
        rightCE->setSigned(true);
      }

      if(isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
        auto leftCE = dyn_cast<ConstantExpr>(left);
        int leftV = leftCE->getZExtValue();
        auto rightCE = dyn_cast<ConstantExpr>(right);
        int rightV = rightCE->getZExtValue();

        if (std::abs(rightV-leftV) <= loopSymMax) {
          auto it = state.loopInfoMap.find(currentBlock);
          if (it != state.loopInfoMap.end() && it->second != nullptr) {
            it->second->indexExpr = nullptr;
          }
          
          ref<Expr> result = SleExpr::create(left, right);
          bindLocal(ki, state, result);
          break;
        }
      }
      
      if (needSymbolic) {
        auto inductionRes = findInductionVariable(state, i, loopInfo);
        if (inductionRes.second) {
          const MemoryObject *mo = inductionRes.second;
          ref<Expr> result;
          ref<Expr> indexExpr = symbolizeIndex(state, loopInfo->indexName, mo, true);
          indexExpr = SExtExpr::create(indexExpr, right->getWidth());
          if (inductionRes.first==0){
            loopInfo->initialVal = left;
            loopInfo->exitExpr = right;
            loopInfo->isSigned = true;
            result = SleExpr::create(indexExpr, right);
            addConstraint(state, SgeExpr::create(indexExpr, left));
          } else {
            loopInfo->initialVal = right;
            loopInfo->exitExpr = left;
            loopInfo->isSigned = true;
            result = SleExpr::create(left, indexExpr);
            addConstraint(state, SleExpr::create(indexExpr, right));
          }

          loopInfo->indexExpr = indexExpr;
          loopInfo->address = mo->getBaseExpr();
          bindLocal(ki, state, result);
          break;
        }
      }
      
      ref<Expr> result = SleExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    default:
      terminateStateOnExecError(state, "invalid ICmp predicate");
    }
    break;
  }
 
    // Memory instructions...
  case Instruction::Alloca: {
    AllocaInst *ai = cast<AllocaInst>(i);
    unsigned elementSize = 
      kmodule->targetData->getTypeStoreSize(ai->getAllocatedType());
    ref<Expr> size = Expr::createPointer(elementSize);
    if (ai->isArrayAllocation()) {
      ref<Expr> count = eval(ki, 0, state).value;
      count = Expr::createZExtToPointerWidth(count);
      size = MulExpr::create(size, count);
    }
    executeAlloc(state, size, true, ki);
    break;
  }

  case Instruction::Load: {
    ref<Expr> base = eval(ki, 0, state).value;

    if (auto se = dyn_cast<SelectExpr>(base)) {
      executeMemoryOperation(state, false, se->trueExpr, 0, ki, 0, base);
      ref<Expr> trueExpr = getDestCell(state, ki).value;

      executeMemoryOperation(state, false, se->falseExpr, 0, ki, 0, base);
      ref<Expr> falseExpr = getDestCell(state, ki).value;

      ref<Expr> result = SelectExpr::create(se->cond, trueExpr, falseExpr);
      bindLocal(ki, state, result);
    } else {
      ref<SelectExpr> se2 = SelectExpr::findSelect(base);
      if (se2) {
        std::map< ref<Expr>, ref<Expr> > replacements;
        replacements[se2] = se2->trueExpr;
        ref<Expr> trueAddress = ConstraintManager::replaceReadExpr(base, replacements);
        executeMemoryOperation(state, false, trueAddress, 0, ki, 0, base);
        ref<Expr> trueRes = getDestCell(state, ki).value;

        replacements.clear();
        replacements[se2] = se2->falseExpr;
        ref<Expr> falseAddress = ConstraintManager::replaceReadExpr(base, replacements);
        executeMemoryOperation(state, false, falseAddress, 0, ki, 0, base);
        ref<Expr> falseRes = getDestCell(state, ki).value;

        ref<Expr> result = SelectExpr::create(se2->cond, trueRes, falseRes);
        bindLocal(ki, state, result);
      } else
        executeMemoryOperation(state, false, base, 0, ki);
    }
    break;
  }
  case Instruction::Store: {    
    ref<Expr> base = eval(ki, 1, state).value;
    ref<Expr> value = eval(ki, 0, state).value;
    llvm::Type *valType = ki->inst->getOperand(0)->getType();
    if (valType->isPointerTy() && isLoopCheck) {
      break;
    }

    if (auto *SI = dyn_cast<StoreInst>(i)) {
      std::string exprStr;
      llvm::raw_string_ostream os(exprStr);
      os << value;
      os.flush(); // optional, os.str() also works
      bool hasTemp = exprStr.find("temp_storage") != std::string::npos;

      if (!hasTemp) {
        Value *ptr = SI->getPointerOperand();
        std::string varSign = getMD(ptr, "var.signedness");
        if (varSign == "signed") {
          if (auto mulExpr = dyn_cast<MulExpr>(value)) {
            if (!mulExpr->isValueSigned() && mulExpr->getWidth()<=32) {
              uint64_t maxValue = (1ULL << (mulExpr->getWidth() - 1)) - 1; 
              ref<Expr> maxValueExpr = ConstantExpr::create(maxValue, Expr::Int64);
              ref<Expr> realRes = MulExpr::create(SExtExpr::create(mulExpr->left, Expr::Int64), SExtExpr::create(mulExpr->right, Expr::Int64));
              ref<Expr> overflowCheck = SleExpr::create(realRes, maxValueExpr);
              StatePair branches = fork(state, overflowCheck, true, BranchType::Trunc);
              ExecutionState *unbound = branches.second;
              if(unbound) {
                terminateStateOnProgramError(*unbound, "integer overflow", StateTerminationType::Overflow);
              }
            }
          }
          if (auto addExpr = dyn_cast<AddExpr>(value)) {
            if (!addExpr->isValueSigned() && addExpr->getWidth()<=32) {
              uint64_t maxValue = (1ULL << (addExpr->getWidth() - 1)) - 1; 
              ref<Expr>  maxValueExpr = ConstantExpr::create(maxValue, Expr::Int64);
              ref<Expr> realRes = AddExpr::create(SExtExpr::create(addExpr->left, Expr::Int64), SExtExpr::create(addExpr->right, Expr::Int64));
              ref<Expr> overflowCheck = SleExpr::create(realRes, maxValueExpr);
              StatePair branches = fork(state, overflowCheck, true, BranchType::Trunc);
              ExecutionState *unbound = branches.second;
              if(unbound) {
                terminateStateOnProgramError(*unbound, "integer overflow", StateTerminationType::Overflow);
              }
            }
          }
        }
      }
    }

    if (valType->isPointerTy()) {
      ref<Expr> value_base_address;
      if (isa<ConstantExpr>(value)) {
        value_base_address = value;
      } else {
        auto vit = state.base_addrs.find(value);
        if (vit != state.base_addrs.end()) {
          value_base_address = vit->second;
        } else {
          value_base_address = Expr::getBaseAddress(value);
        }
      }
      if (!value_base_address.isNull()) {
        ObjectPair op0;
        bool success0;
        if (state.addressSpace.resolveOne(state, solver.get(), value_base_address, op0, success0) && success0) {
          ref<Expr> init_base = op0.first->getBaseExpr();
          if (sharedAddresses.find(init_base) != sharedAddresses.end()) {
            std::string symName = sharedAddresses[init_base].first;
            if (sharedAddresses[init_base].second) {
              if (op0.first->name.empty() || op0.first->name!=symName || state.symbolicArrayMap.find(symName)==state.symbolicArrayMap.end() || !op0.second->hasArray()) {
                op0.first->setName(symName);
                ref<Expr> memSizeExpr = state.kernelConfig.getSharedMemSize();
                int size = op0.first->size;
                if (auto memSizeCE = dyn_cast<ConstantExpr>(memSizeExpr)) {
                  if (memSizeCE->getZExtValue() == 0) {
                    memSizeExpr = ConstantExpr::create(228*1024, memSizeExpr->getWidth());
                  } else {
                    size = memSizeCE->getZExtValue();
                  }
                }

                const Array *array = arrayCache.CreateArray(symName, size);
                bindObjectInState(state, op0.first, false, array);
                state.addSymbolic(op0.first, array);

                SymArrayMemoryObject *symmo = new SymArrayMemoryObject(op0.first->address, array, nullptr, memSizeExpr);
                state.symbolicArrayMap[symName] = symmo;
              }
            }
          }
        }
      }
    }
    
    executeMemoryOperation(state, true, base, value, ki);
    break;
  }

  case Instruction::GetElementPtr: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);
    ref<Expr> base = eval(ki, 0, state).value;
    ref<Expr> original_base = base;

    ref<Expr> mo_base_address;
    if (isa<ConstantExpr>(base)) {
      mo_base_address = base;
    } else {
      auto base_it = state.base_addrs.find(base);
      if (base_it != state.base_addrs.end()) {
        mo_base_address = base_it->second;
      } else {
        mo_base_address = Expr::getBaseAddress(base);
      }
    }
    if (!mo_base_address.isNull()) {
      ObjectPair op0;
      bool success0;
      if (state.addressSpace.resolveOne(state, solver.get(), mo_base_address, op0, success0) && success0) {
        ref<Expr> init_base = op0.first->getBaseExpr();
        if (sharedAddresses.find(init_base) != sharedAddresses.end()) {
          std::string symName = sharedAddresses[init_base].first;
          if (sharedAddresses[init_base].second) {
            if (op0.first->name.empty() || op0.first->name!=symName || state.symbolicArrayMap.find(symName)==state.symbolicArrayMap.end() || !op0.second->hasArray()) {
              op0.first->setName(symName);
              ref<Expr> memSizeExpr = state.kernelConfig.getSharedMemSize();
              int size = op0.first->size;
              if (auto memSizeCE = dyn_cast<ConstantExpr>(memSizeExpr)) {
                if (memSizeCE->getZExtValue() == 0) {
                  memSizeExpr = ConstantExpr::create(228*1024, memSizeExpr->getWidth());
                } else {
                  size = memSizeCE->getZExtValue();
                }
              }

              const Array *array = arrayCache.CreateArray(symName, size);
              bindObjectInState(state, op0.first, false, array);
              state.addSymbolic(op0.first, array);

              SymArrayMemoryObject *symmo = new SymArrayMemoryObject(op0.first->address, array, nullptr, memSizeExpr);
              state.symbolicArrayMap[symName] = symmo;

              if (auto *gep = dyn_cast<GetElementPtrInst>(i)) {
                Type *ptrTy = gep->getPointerOperandType(); 
                Type *elemTy = ptrTy->getPointerElementType(); 

                if (auto *arrTy = dyn_cast<ArrayType>(elemTy)) {
                  Type *arrElemTy = arrTy->getElementType();
                  if (arrElemTy->isFloatingPointTy()) {
                    symmo->isFloat = true;
                  }
                }
              }
            }
          }
        }
      }
    }

    for (std::vector< std::pair<unsigned, uint64_t> >::iterator 
           it = kgepi->indices.begin(), ie = kgepi->indices.end(); 
         it != ie; ++it) {
      uint64_t elementSize = it->second;
      ref<Expr> index = eval(ki, it->first, state).value;
      base = AddExpr::create(base,
                             MulExpr::create(Expr::createSExtToPointerWidth(index),
                                             Expr::createPointer(elementSize)));
    }
    if (kgepi->offset)
      base = AddExpr::create(base,
                             Expr::createPointer(kgepi->offset));

    if (SingleObjectResolution) {
      if (isa<ConstantExpr>(original_base) && !isa<ConstantExpr>(base)) {
        // the initial base address was a constant expression, the final is not:
        // store the mapping between constant address and the non-const
        // reference in the state
        ref<ConstantExpr> c_orig_base = dyn_cast<ConstantExpr>(original_base);

        ObjectPair op;
        if (state.addressSpace.resolveOne(c_orig_base, op)) {
          // store the address of the MemoryObject associated with this GEP
          // instruction
          state.base_mos[op.first->address].insert(base);
          ref<ConstantExpr> r =
              ConstantExpr::alloc(op.first->address, Expr::Int64);
          state.base_addrs[base] = r;
        } else {
          // this case should not happen - we have a GEP instruction with const
          // base address, so we should be able to find an exact memory object
          // match
          klee_warning("Failed to find a memory object for address %" PRIx64,
                       c_orig_base->getZExtValue());
        }

      } else if (!isa<ConstantExpr>(original_base)) {
        auto base_it = state.base_addrs.find(original_base);
        if (base_it != state.base_addrs.end()) {
          // we need to update the current entry with a new value
          uint64_t address = base_it->second->getZExtValue();
          state.base_mos[address].insert(base);
          state.base_addrs[base] = base_it->second;
        }
      }
    }

    bindLocal(ki, state, base);
    break;
  }

    // Conversion
  case Instruction::Trunc: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> srcValue = eval(ki, 0, state).value;
    ref<Expr> result = ExtractExpr::create(srcValue,
                                           0,
                                           getWidthForLLVMType(ci->getType()));
    Type *srcType = ci->getSrcTy();
    Type *destType = ci->getDestTy();

    if (!srcType->isIntegerTy() || !destType->isIntegerTy() || destType->getIntegerBitWidth() >= 64 || isa<ConstantExpr>(srcValue) || isLoopCheck || srcType->getIntegerBitWidth() >= 64) {
      bindLocal(ki, state, result);
      break; // Not an integer truncation, no need to check for overflow
    }

    unsigned destBitWidth = destType->getIntegerBitWidth();
    ref<Expr> minValueCheck;
    ref<Expr> maxValueCheck;
    ref<ReadExpr> readArg = ReadExpr::extractReadExpr(srcValue);
    if (readArg && readArg->updates.root && !readArg->updates.root->name.empty()) {
      std::string argName = readArg->updates.root->name;
      if (isIntArgSigned.find(argName) != isIntArgSigned.end()) {
        if (isIntArgSigned[argName]) {
          // if (destBitWidth <= 64) {
            // ref<ConstantExpr> minValueExpr = ConstantExpr::alloc(-(1LL << (destBitWidth-1)), srcValue->getWidth());   
            ref<Expr> maxValueExpr = ConstantExpr::create((1ULL << (destBitWidth-1)) - 1, srcValue->getWidth());
            // minValueCheck = SgeExpr::create(srcValue, minValueExpr);
            maxValueCheck = SleExpr::create(srcValue, maxValueExpr);
            minValueCheck = ConstantExpr::create(1, Expr::Bool);
        } else {
          ref<Expr> minValueExpr = ConstantExpr::create(0, srcValue->getWidth());
          ref<Expr> maxValueExpr = ConstantExpr::create((1ULL << destBitWidth) - 1, srcValue->getWidth()); 
          minValueCheck = UgeExpr::create(srcValue, minValueExpr);
          maxValueCheck = UleExpr::create(srcValue, maxValueExpr);
        }
      }
    }

    ref<Expr> overflowCheck;
    if(!minValueCheck || !maxValueCheck) {
      if (destBitWidth == 1)
        overflowCheck = AndExpr::create(UgeExpr::create(srcValue, ConstantExpr::create(0, srcValue->getWidth())), UleExpr::create(srcValue, ConstantExpr::create(1, srcValue->getWidth())));
      else
        overflowCheck = UleExpr::create(srcValue, ConstantExpr::create((1ULL << destBitWidth) - 1, srcValue->getWidth()));
    } else
      overflowCheck = AndExpr::create(minValueCheck, maxValueCheck);

    StatePair branches = fork(state, overflowCheck, true, BranchType::Trunc);
    ExecutionState *bound = branches.first;
    if (bound) {
      bindLocal(ki, state, result);
    }

    ExecutionState *unbound = branches.second;
    if(unbound) {
      terminateStateOnProgramError(*unbound, "integer overflow", StateTerminationType::Overflow);
    }

    break;
  }
  case Instruction::ZExt: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = ZExtExpr::create(eval(ki, 0, state).value,
                                        getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::SExt: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = SExtExpr::create(eval(ki, 0, state).value,
                                        getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::IntToPtr: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width pType = getWidthForLLVMType(ci->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    bindLocal(ki, state, ZExtExpr::create(arg, pType));
    break;
  }
  case Instruction::PtrToInt: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width iType = getWidthForLLVMType(ci->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    bindLocal(ki, state, ZExtExpr::create(arg, iType));
    break;
  }

  case Instruction::BitCast: {
    ref<Expr> result = eval(ki, 0, state).value;
    bindLocal(ki, state, result);


    llvm::BitCastInst *BCI = cast<llvm::BitCastInst>(i);
    bool success;
    ObjectPair op;
    if (isa<ConstantExpr>(result) && state.addressSpace.resolveOne(state, solver.get(), result, op, success) && success) {
      const MemoryObject *mo = op.first;
      auto resultCE = dyn_cast<ConstantExpr>(result);
      if (mo->address != resultCE->getZExtValue()) {
        break;
      }

      if (sharedAddresses.find(mo->getBaseExpr()) != sharedAddresses.end() && sharedAddresses[mo->getBaseExpr()].second) {
        break;
      }
    
      if (state.symbolicArrayMap.find(mo->name) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[mo->name];
        if (!symmo->elementSize.isNull()) {
          break;
        }

        llvm::Type *targetType = BCI->getDestTy(); 
        if (targetType->isPointerTy()) {
            targetType = targetType->getPointerElementType();
        }
        uint64_t elementSize = getTypeBits(targetType) / 8; // bytes
        if (!elementSize) {
          if (auto *structType = llvm::dyn_cast<llvm::StructType>(targetType)) {
            if (structType->getName().find("at::TensorBase")!=std::string::npos) {
              break;
            }
            if (structType->isSized()) {
              elementSize = kmodule->targetData->getTypeAllocSize(structType);
            }
          }
        }
        if (elementSize) {
          ref<ConstantExpr> elementSizeExpr = ConstantExpr::create(elementSize, Expr::Int64, false);
          symmo->elementSize = elementSizeExpr;
        }
      }
    }
    break;
  }

    // Floating point instructions
  case Instruction::FNeg: {
    ref<Expr> arg = eval(ki, 0, state).value;
    if (!fpWidthToSemantics(arg->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FNeg operation");

    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                        "floating point");
    
      llvm::APFloat Res(*fpWidthToSemantics(arg_constant->getWidth()), arg_constant->getAPValue());
      Res = llvm::neg(Res);
      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
    } else {
      bindLocal(ki, state, SubExpr::create(ConstantExpr::create(0, arg->getWidth()), arg));
    }
    break;
  }

  case Instruction::FAdd: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FAdd operation");

    if (isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
      ref<ConstantExpr> left_constant = toConstant(state, left,
                                          "floating point");
      ref<ConstantExpr> right_constant = toConstant(state, right,
                                          "floating point");

      llvm::APFloat Res(*fpWidthToSemantics(left_constant->getWidth()), left_constant->getAPValue());
      Res.add(APFloat(*fpWidthToSemantics(right_constant->getWidth()),right_constant->getAPValue()), APFloat::rmNearestTiesToEven);
      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(left)) {
        CE->setIsFloat(true);
      }
      if (auto CE = dyn_cast<ConstantExpr>(right)) {
        CE->setIsFloat(true);
      }
      bindLocal(ki, state, AddExpr::create(left, right));
    }
    break;
  }

  case Instruction::FSub: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FSub operation");

    if (isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
      ref<ConstantExpr> left_constant = toConstant(state, left,
                                          "floating point");
      ref<ConstantExpr> right_constant = toConstant(state, right,
                                          "floating point");

      llvm::APFloat Res(*fpWidthToSemantics(left_constant->getWidth()), left_constant->getAPValue());
      Res.subtract(APFloat(*fpWidthToSemantics(right_constant->getWidth()), right_constant->getAPValue()), APFloat::rmNearestTiesToEven);
      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(left)) {
        CE->setIsFloat(true);
      }
      if (auto CE = dyn_cast<ConstantExpr>(right)) {
        CE->setIsFloat(true);
      }
      bindLocal(ki, state, SubExpr::create(left, right));
    }
    break;
  }

  case Instruction::FMul: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FMul operation");

   if (isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
      ref<ConstantExpr> left_constant = toConstant(state, left,
                                          "floating point");
      ref<ConstantExpr> right_constant = toConstant(state, right,
                                          "floating point");

      llvm::APFloat Res(*fpWidthToSemantics(left_constant->getWidth()), left_constant->getAPValue());
      Res.multiply(APFloat(*fpWidthToSemantics(right_constant->getWidth()), right_constant->getAPValue()), APFloat::rmNearestTiesToEven);
      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(left)) {
        CE->setIsFloat(true);
      }
      if (auto CE = dyn_cast<ConstantExpr>(right)) {
        CE->setIsFloat(true);
      }
      bindLocal(ki, state, MulExpr::create(left, right));
    }
    break;
  }

  case Instruction::FDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FDiv operation");

    if (isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
      ref<ConstantExpr> left_constant = toConstant(state, left,
                                          "floating point");
      ref<ConstantExpr> right_constant = toConstant(state, right,
                                          "floating point");

      llvm::APFloat Res(*fpWidthToSemantics(left_constant->getWidth()), left_constant->getAPValue());
      Res.divide(APFloat(*fpWidthToSemantics(right_constant->getWidth()), right_constant->getAPValue()), APFloat::rmNearestTiesToEven);
      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(left)) {
        CE->setIsFloat(true);
      }
      if (auto CE = dyn_cast<ConstantExpr>(right)) {
        CE->setIsFloat(true);
      }
      ref<Expr> cond = NeExpr::create(right, ConstantExpr::create(0, right->getWidth()));

      if (isLoopCheck) {
        bindLocal(ki, state, SDivExpr::create(left, right));
      } else {
        StatePair branches = fork(state, cond, true, BranchType::Conditional);
        if (branches.second) {
          terminateStateOnProgramError(*branches.second, "divisor of fdiv is zero", StateTerminationType::Assert);
        }
        if (branches.first)
          bindLocal(ki, *branches.first, SDivExpr::create(left, right));
      }
    }
    break;
  }

  case Instruction::FRem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FRem operation");

    if (isa<ConstantExpr>(left) && isa<ConstantExpr>(right)) {
      ref<ConstantExpr> left_constant = toConstant(state, left,
                                          "floating point");
      ref<ConstantExpr> right_constant = toConstant(state, right,
                                          "floating point");

      llvm::APFloat Res(*fpWidthToSemantics(left_constant->getWidth()), left_constant->getAPValue());
      Res.mod(
          APFloat(*fpWidthToSemantics(right_constant->getWidth()), right_constant->getAPValue()));
      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt(), true, true));
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(left)) {
        CE->setIsFloat(true);
      }
      if (auto CE = dyn_cast<ConstantExpr>(right)) {
        CE->setIsFloat(true);
      }
      bindLocal(ki, state, SRemExpr::create(left, right));
    }
    break;
  }

  case Instruction::FPTrunc: {
    FPTruncInst *fi = cast<FPTruncInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > arg->getWidth())
      return terminateStateOnExecError(state, "Unsupported FPTrunc operation");

    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                        "floating point");

      llvm::APFloat Res(*fpWidthToSemantics(arg_constant->getWidth()), arg_constant->getAPValue());
      bool losesInfo = false;
      Res.convert(*fpWidthToSemantics(resultType),
                  llvm::APFloat::rmNearestTiesToEven,
                  &losesInfo);
      bindLocal(ki, state, ConstantExpr::alloc(Res, true, true));
    } else {
      bindLocal(ki, state, ZExtExpr::create(arg, resultType));
    }
    break;
  }

  case Instruction::FPExt: {
    FPExtInst *fi = cast<FPExtInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    if (!fpWidthToSemantics(arg->getWidth()) || arg->getWidth() > resultType)
      return terminateStateOnExecError(state, "Unsupported FPExt operation");
    
    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                        "floating point");
    
      llvm::APFloat Res(*fpWidthToSemantics(arg_constant->getWidth()), arg_constant->getAPValue());
      bool losesInfo = false;
      Res.convert(*fpWidthToSemantics(resultType),
                  llvm::APFloat::rmNearestTiesToEven,
                  &losesInfo);
      bindLocal(ki, state, ConstantExpr::alloc(Res, true, true));
    } else {
      bindLocal(ki, state, ZExtExpr::create(arg, resultType));
    }
    break;
  }

  case Instruction::FPToUI: {
    FPToUIInst *fi = cast<FPToUIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64)
      return terminateStateOnExecError(state, "Unsupported FPToUI operation");

    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                       "floating point");
      llvm::APFloat Arg(*fpWidthToSemantics(arg_constant->getWidth()), arg_constant->getAPValue());
      uint64_t value = 0;
      bool isExact = true;
#if LLVM_VERSION_CODE >= LLVM_VERSION(16, 0)
      auto valueRef = llvm::MutableArrayRef(value);
#else
      auto valueRef = makeMutableArrayRef(value);
#endif
      Arg.convertToInteger(valueRef, resultType, false,
                          llvm::APFloat::rmTowardZero, &isExact);
      bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    } else {
      bindLocal(ki, state, ZExtExpr::create(arg, resultType));
    }
    break;
  }

  case Instruction::FPToSI: {
    FPToSIInst *fi = cast<FPToSIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64)
      return terminateStateOnExecError(state, "Unsupported FPToSI operation");

    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                       "floating point");
      llvm::APFloat Arg(*fpWidthToSemantics(arg_constant->getWidth()), arg_constant->getAPValue());

      uint64_t value = 0;
      bool isExact = true;
#if LLVM_VERSION_CODE >= LLVM_VERSION(16, 0)
      auto valueRef = llvm::MutableArrayRef(value);
#else
      auto valueRef = makeMutableArrayRef(value);
#endif
      Arg.convertToInteger(valueRef, resultType, true,
                          llvm::APFloat::rmTowardZero, &isExact);
      bindLocal(ki, state, ConstantExpr::alloc(value, resultType, true));
    } else {
      bindLocal(ki, state, SExtExpr::create(arg, resultType));
    }
    break;
  }

  case Instruction::UIToFP: {
    UIToFPInst *fi = cast<UIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
      if (!semantics)
        return terminateStateOnExecError(state, "Unsupported UIToFP operation");

    ref<Expr> arg = eval(ki, 0, state).value;
    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                       "floating point");
      
      llvm::APFloat f(*semantics, 0);
      f.convertFromAPInt(arg_constant->getAPValue(), false,
                        llvm::APFloat::rmNearestTiesToEven);

      bindLocal(ki, state, ConstantExpr::alloc(f, false, true));
    } else {
      bindLocal(ki, state, ZExtExpr::create(arg, resultType));
    }
    break;
  }

  case Instruction::SIToFP: {
    SIToFPInst *fi = cast<SIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics)
      return terminateStateOnExecError(state, "Unsupported SIToFP operation");

    ref<Expr> arg = eval(ki, 0, state).value;
    if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                       "floating point");
      
      llvm::APFloat f(*semantics, 0);
      f.convertFromAPInt(arg_constant->getAPValue(), true,
                        llvm::APFloat::rmNearestTiesToEven);

      bindLocal(ki, state, ConstantExpr::alloc(f, true, true));
    } else {
      bindLocal(ki, state, SExtExpr::create(arg, resultType));
    }
    break;
  }

  case Instruction::FCmp: {
    FCmpInst *fi = cast<FCmpInst>(i);
    ref<Expr> firstArg = eval(ki, 0, state).value;
    ref<Expr> secondArg = eval(ki, 1, state).value;

    if (isa<ConstantExpr>(firstArg) && isa<ConstantExpr>(secondArg)) {
      ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                          "floating point");
      ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                          "floating point");
      if (!fpWidthToSemantics(left->getWidth()) ||
          !fpWidthToSemantics(right->getWidth()))
        return terminateStateOnExecError(state, "Unsupported FCmp operation");

      APFloat LHS(*fpWidthToSemantics(left->getWidth()),left->getAPValue());
      APFloat RHS(*fpWidthToSemantics(right->getWidth()),right->getAPValue());
      APFloat::cmpResult CmpRes = LHS.compare(RHS);

      bool Result = false;
      switch( fi->getPredicate() ) {
        // Predicates which only care about whether or not the operands are NaNs.
      case FCmpInst::FCMP_ORD:
        Result = (CmpRes != APFloat::cmpUnordered);
        break;

      case FCmpInst::FCMP_UNO:
        Result = (CmpRes == APFloat::cmpUnordered);
        break;

        // Ordered comparisons return false if either operand is NaN.  Unordered
        // comparisons return true if either operand is NaN.
      case FCmpInst::FCMP_UEQ:
        Result = (CmpRes == APFloat::cmpUnordered || CmpRes == APFloat::cmpEqual);
        break;
      case FCmpInst::FCMP_OEQ:
        Result = (CmpRes != APFloat::cmpUnordered && CmpRes == APFloat::cmpEqual);
        break;

      case FCmpInst::FCMP_UGT:
        Result = (CmpRes == APFloat::cmpUnordered || CmpRes == APFloat::cmpGreaterThan);
        break;
      case FCmpInst::FCMP_OGT:
        Result = (CmpRes != APFloat::cmpUnordered && CmpRes == APFloat::cmpGreaterThan);
        break;

      case FCmpInst::FCMP_UGE:
        Result = (CmpRes == APFloat::cmpUnordered || (CmpRes == APFloat::cmpGreaterThan || CmpRes == APFloat::cmpEqual));
        break;
      case FCmpInst::FCMP_OGE:
        Result = (CmpRes != APFloat::cmpUnordered && (CmpRes == APFloat::cmpGreaterThan || CmpRes == APFloat::cmpEqual));
        break;

      case FCmpInst::FCMP_ULT:
        Result = (CmpRes == APFloat::cmpUnordered || CmpRes == APFloat::cmpLessThan);
        break;
      case FCmpInst::FCMP_OLT:
        Result = (CmpRes != APFloat::cmpUnordered && CmpRes == APFloat::cmpLessThan);
        break;

      case FCmpInst::FCMP_ULE:
        Result = (CmpRes == APFloat::cmpUnordered || (CmpRes == APFloat::cmpLessThan || CmpRes == APFloat::cmpEqual));
        break;
      case FCmpInst::FCMP_OLE:
        Result = (CmpRes != APFloat::cmpUnordered && (CmpRes == APFloat::cmpLessThan || CmpRes == APFloat::cmpEqual));
        break;

      case FCmpInst::FCMP_UNE:
        Result = (CmpRes == APFloat::cmpUnordered || CmpRes != APFloat::cmpEqual);
        break;
      case FCmpInst::FCMP_ONE:
        Result = (CmpRes != APFloat::cmpUnordered && CmpRes != APFloat::cmpEqual);
        break;

      default:
        assert(0 && "Invalid FCMP predicate!");
        break;
      case FCmpInst::FCMP_FALSE:
        Result = false;
        break;
      case FCmpInst::FCMP_TRUE:
        Result = true;
        break;
      }

      bindLocal(ki, state, ConstantExpr::alloc(Result, Expr::Bool));
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(firstArg)) {
        CE->setIsFloat(true);
      }
      if (auto CE = dyn_cast<ConstantExpr>(secondArg)) {
        CE->setIsFloat(true);
      }
      ref<Expr> result;
      switch(fi->getPredicate()) {
      case FCmpInst::FCMP_ORD:
        result = ConstantExpr::create(1, Expr::Bool);
        break;

      case FCmpInst::FCMP_UNO:
        result = ConstantExpr::create(1, Expr::Bool);
        break;

      case FCmpInst::FCMP_UEQ:
        result = EqExpr::create(firstArg, secondArg);
        break;
      case FCmpInst::FCMP_OEQ:
        result = EqExpr::create(firstArg, secondArg);
        break;

      case FCmpInst::FCMP_UGT:
        result = UgtExpr::create(firstArg, secondArg);
        break;
      case FCmpInst::FCMP_OGT:
        result = SgtExpr::create(firstArg, secondArg);
        break;

      case FCmpInst::FCMP_UGE:
        result = UgeExpr::create(firstArg, secondArg);
        break;
      case FCmpInst::FCMP_OGE:
        result = SgeExpr::create(firstArg, secondArg);
        break;

      case FCmpInst::FCMP_ULT:
        result = UltExpr::create(firstArg, secondArg);
        break;
      case FCmpInst::FCMP_OLT:
        result = SltExpr::create(firstArg, secondArg);
        break;

      case FCmpInst::FCMP_ULE:
        result = UleExpr::create(firstArg, secondArg);
        break;
      case FCmpInst::FCMP_OLE:
        result = SleExpr::create(firstArg, secondArg);
        break;

      case FCmpInst::FCMP_UNE:
        result = NeExpr::create(firstArg, secondArg);
        break;
      case FCmpInst::FCMP_ONE:
        result = NeExpr::create(firstArg, secondArg);
        break;

      default:
        assert(0 && "Invalid FCMP predicate!");
        break;
      case FCmpInst::FCMP_FALSE:
        result = ConstantExpr::create(0, Expr::Bool);
        break;
      case FCmpInst::FCMP_TRUE:
        result = ConstantExpr::create(1, Expr::Bool);
        break;
      }
      bindLocal(ki, state, result);
    }
    break;
  }
  case Instruction::InsertValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    ref<Expr> agg = eval(ki, 0, state).value;
    ref<Expr> val = eval(ki, 1, state).value;

    ref<Expr> l = NULL, r = NULL;
    unsigned lOffset = kgepi->offset*8, rOffset = kgepi->offset*8 + val->getWidth();

    if (lOffset > 0)
      l = ExtractExpr::create(agg, 0, lOffset);
    if (rOffset < agg->getWidth())
      r = ExtractExpr::create(agg, rOffset, agg->getWidth() - rOffset);

    ref<Expr> result;
    if (l && r)
      result = ConcatExpr::create(r, ConcatExpr::create(val, l));
    else if (l)
      result = ConcatExpr::create(val, l);
    else if (r)
      result = ConcatExpr::create(r, val);
    else
      result = val;

    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ExtractValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    ref<Expr> agg = eval(ki, 0, state).value;

    ref<Expr> result = ExtractExpr::create(agg, kgepi->offset*8, getWidthForLLVMType(i->getType()));

    bindLocal(ki, state, result);
    break;
  }
  case Instruction::Fence: {
    // Ignore for now
    break;
  }
  case Instruction::InsertElement: {
    InsertElementInst *iei = cast<InsertElementInst>(i);
    ref<Expr> vec = eval(ki, 0, state).value;
    ref<Expr> newElt = eval(ki, 1, state).value;
    ref<Expr> idx = eval(ki, 2, state).value;

    ConstantExpr *cIdx = dyn_cast<ConstantExpr>(idx);
    if (cIdx == NULL) {
      terminateStateOnExecError(
          state, "InsertElement, support for symbolic index not implemented");
      return;
    }
    uint64_t iIdx = cIdx->getZExtValue();
    const auto *vt = cast<llvm::FixedVectorType>(iei->getType());
    unsigned EltBits = getWidthForLLVMType(vt->getElementType());

    if (iIdx >= vt->getNumElements()) {
      // Out of bounds write
      terminateStateOnProgramError(state,
                                   "Out of bounds write when inserting element",
                                   StateTerminationType::BadVectorAccess);
      return;
    }

    const unsigned elementCount = vt->getNumElements();
    llvm::SmallVector<ref<Expr>, 8> elems;
    elems.reserve(elementCount);
    for (unsigned i = elementCount; i != 0; --i) {
      auto of = i - 1;
      unsigned bitOffset = EltBits * of;
      elems.push_back(
          of == iIdx ? newElt : ExtractExpr::create(vec, bitOffset, EltBits));
    }

    assert(Context::get().isLittleEndian() && "FIXME:Broken for big endian");
    ref<Expr> Result = ConcatExpr::createN(elementCount, elems.data());
    bindLocal(ki, state, Result);
    break;
  }
  case Instruction::ExtractElement: {
    ExtractElementInst *eei = cast<ExtractElementInst>(i);
    ref<Expr> vec = eval(ki, 0, state).value;
    ref<Expr> idx = eval(ki, 1, state).value;

    ConstantExpr *cIdx = dyn_cast<ConstantExpr>(idx);
    if (cIdx == NULL) {
      terminateStateOnExecError(
          state, "ExtractElement, support for symbolic index not implemented");
      return;
    }
    uint64_t iIdx = cIdx->getZExtValue();
    const auto *vt = cast<llvm::FixedVectorType>(eei->getVectorOperandType());
    unsigned EltBits = getWidthForLLVMType(vt->getElementType());

    if (iIdx >= vt->getNumElements()) {
      // Out of bounds read
      terminateStateOnProgramError(state,
                                   "Out of bounds read when extracting element",
                                   StateTerminationType::BadVectorAccess);
      return;
    }

    unsigned bitOffset = EltBits * iIdx;
    ref<Expr> Result = ExtractExpr::create(vec, bitOffset, EltBits);
    bindLocal(ki, state, Result);
    break;
  }
  case Instruction::ShuffleVector:
    // Should never happen due to Scalarizer pass removing ShuffleVector
    // instructions.
    terminateStateOnExecError(state, "Unexpected ShuffleVector instruction");
    break;

#ifdef SUPPORT_KLEE_EH_CXX
  case Instruction::Resume: {
    auto *cui = dyn_cast_or_null<CleanupPhaseUnwindingInformation>(
        state.unwindingInformation.get());

    if (!cui) {
      terminateStateOnExecError(
          state,
          "resume-instruction executed outside of cleanup phase unwinding");
      break;
    }

    ref<Expr> arg = eval(ki, 0, state).value;
    ref<Expr> exceptionPointer = ExtractExpr::create(arg, 0, Expr::Int64);
    ref<Expr> selectorValue =
        ExtractExpr::create(arg, Expr::Int64, Expr::Int32);

    if (!dyn_cast<ConstantExpr>(exceptionPointer) ||
        !dyn_cast<ConstantExpr>(selectorValue)) {
      terminateStateOnExecError(
          state, "resume-instruction called with non constant expression");
      break;
    }

    if (!Expr::createIsZero(selectorValue)->isTrue()) {
      klee_warning("resume-instruction called with non-0 selector value");
    }

    if (!EqExpr::create(exceptionPointer, cui->exceptionObject)->isTrue()) {
      terminateStateOnExecError(
          state, "resume-instruction called with unexpected exception pointer");
      break;
    }

    unwindToNextLandingpad(state);
    break;
  }

  case Instruction::LandingPad: {
    auto *cui = dyn_cast_or_null<CleanupPhaseUnwindingInformation>(
        state.unwindingInformation.get());

    if (!cui) {
      terminateStateOnExecError(
          state, "Executing landing pad but not in unwinding phase 2");
      break;
    }

    ref<ConstantExpr> exceptionPointer = cui->exceptionObject;
    ref<ConstantExpr> selectorValue;

    // check on which frame we are currently
    if (state.stack.size() - 1 == cui->catchingStackIndex) {
      // we are in the target stack frame, return the selector value
      // that was returned by the personality fn in phase 1 and stop unwinding.
      selectorValue = cui->selectorValue;

      // stop unwinding by cleaning up our unwinding information.
      state.unwindingInformation.reset();

      // this would otherwise now be a dangling pointer
      cui = nullptr;
    } else {
      // we are not yet at the target stack frame. the landingpad might have
      // a cleanup clause or not, anyway, we give it the selector value "0",
      // which represents a cleanup, and expect it to handle it.
      // This is explicitly allowed by LLVM, see
      // https://llvm.org/docs/ExceptionHandling.html#id18
      selectorValue = ConstantExpr::create(0, Expr::Int32);
    }

    ref<Expr> result = ConcatExpr::create(
        ZExtExpr::create(selectorValue, Expr::Int32), exceptionPointer);

    bindLocal(ki, state, result);

    break;
  }
#endif // SUPPORT_KLEE_EH_CXX

  case Instruction::AtomicRMW:
    terminateStateOnExecError(state, "Unexpected Atomic instruction, should be "
                                     "lowered by LowerAtomicInstructionPass");
    break;
  case Instruction::AtomicCmpXchg:
    terminateStateOnExecError(state,
                              "Unexpected AtomicCmpXchg instruction, should be "
                              "lowered by LowerAtomicInstructionPass");
    break;
  // Other instructions...
  // Unhandled
  default:
    terminateStateOnExecError(state, "illegal instruction");
    break;
  }
}

void Executor::updateStates(ExecutionState *current) {
  if (searcher) {
    searcher->update(current, addedStates, removedStates);
  }
  
  states.insert(addedStates.begin(), addedStates.end());
  addedStates.clear();

  for (std::vector<ExecutionState *>::iterator it = removedStates.begin(),
                                               ie = removedStates.end();
       it != ie; ++it) {
    ExecutionState *es = *it;
    std::set<ExecutionState*>::iterator it2 = states.find(es);
    if (it2 == states.end())
      continue;
    states.erase(it2);
    std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(es);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    executionTree->remove(es->executionTreeNode);
    delete es;
  }
  removedStates.clear();
}

template <typename TypeIt>
void Executor::computeOffsetsSeqTy(KGEPInstruction *kgepi,
                                   ref<ConstantExpr> &constantOffset,
                                   uint64_t index, const TypeIt it) {
#if LLVM_VERSION_CODE <= LLVM_VERSION(14, 0)
  assert(it->getNumContainedTypes() == 1 &&
         "Sequential type must contain one subtype");
  uint64_t elementSize =
      kmodule->targetData->getTypeStoreSize(it->getContainedType(0));
#else
  assert(it.isSequential() && "Called with non-sequential type");
  // Get the size of a single element
  std::uint64_t elementSize =
      kmodule->targetData->getTypeStoreSize(it.getIndexedType());
#endif
  const Value *operand = it.getOperand();
  if (const Constant *c = dyn_cast<Constant>(operand)) {
    ref<ConstantExpr> index =
        evalConstant(c)->SExt(Context::get().getPointerWidth());
    ref<ConstantExpr> addend = index->Mul(
        ConstantExpr::alloc(elementSize, Context::get().getPointerWidth()));
    constantOffset = constantOffset->Add(addend);
  } else {
    kgepi->indices.emplace_back(index, elementSize);
  }
}

template <typename TypeIt>
void Executor::computeOffsets(KGEPInstruction *kgepi, TypeIt ib, TypeIt ie) {
  ref<ConstantExpr> constantOffset =
    ConstantExpr::alloc(0, Context::get().getPointerWidth());
  uint64_t index = 1;
  for (TypeIt ii = ib; ii != ie; ++ii) {
#if LLVM_VERSION_CODE <= LLVM_VERSION(14, 0)
    if (StructType *st = dyn_cast<StructType>(*ii)) {
#else
    if (StructType *st = ii.getStructTypeOrNull()) {
#endif
      const StructLayout *sl = kmodule->targetData->getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(ii.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned) ci->getZExtValue());
      constantOffset = constantOffset->Add(ConstantExpr::alloc(addend,
                                                               Context::get().getPointerWidth()));
#if LLVM_VERSION_CODE <= LLVM_VERSION(14, 0)
    } else if (ii->isArrayTy() || ii->isVectorTy() || ii->isPointerTy()) {
#else
    } else if (ii.isSequential()) {
#endif
      computeOffsetsSeqTy(kgepi, constantOffset, index, ii);
    } else
      assert("invalid type" && 0);
    index++;
  }
  kgepi->offset = constantOffset->getZExtValue();
}

void Executor::bindInstructionConstants(KInstruction *KI) {
  if (GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(KI->inst)) {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction *>(KI);
#if LLVM_VERSION_CODE <= LLVM_VERSION(14, 0)
    computeOffsets(kgepi, klee::gep_type_begin(gepi), klee::gep_type_end(gepi));
#else
    computeOffsets(kgepi, llvm::gep_type_begin(gepi), llvm::gep_type_end(gepi));
#endif
  } else if (InsertValueInst *ivi = dyn_cast<InsertValueInst>(KI->inst)) {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction *>(KI);
    llvm::Value *agg = ivi->getAggregateOperand();
    llvm::Type *current_type = agg->getType();
    std::uint64_t offset = 0;
    for (auto index : ivi->indices()) {
      if (StructType *st = dyn_cast<llvm::StructType>(current_type)) {
        const StructLayout *sl = kmodule->targetData->getStructLayout(st);
        std::uint64_t addend = sl->getElementOffset(index);
        offset = offset + addend;
      } else if (current_type->isArrayTy() || current_type->isVectorTy() ||
                 current_type->isPointerTy()) {
        std::uint64_t elementSize = kmodule->targetData->getTypeStoreSize(
            current_type->getArrayElementType());
        offset += elementSize * index;
      } else {
        assert(0 && "Unknown type");
      }

      current_type = GetElementPtrInst::getTypeAtIndex(current_type, index);
    }
    kgepi->offset = offset;
  } else if (ExtractValueInst *evi = dyn_cast<ExtractValueInst>(KI->inst)) {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction *>(KI);
    llvm::Value *agg = evi->getAggregateOperand();
    llvm::Type *current_type = agg->getType();
    uint64_t offset = 0;
    for (auto index : evi->indices()) {
      if (StructType *st = dyn_cast<llvm::StructType>(current_type)) {
        const StructLayout *sl = kmodule->targetData->getStructLayout(st);
        uint64_t addend = sl->getElementOffset(index);
        offset = offset + addend;
      } else if (current_type->isArrayTy() || current_type->isVectorTy() ||
                 current_type->isPointerTy()) {
        uint64_t elementSize = kmodule->targetData->getTypeStoreSize(
            current_type->getArrayElementType());
        offset += elementSize * index;
      } else {
        assert(0 && "Unknown type");
      }

      current_type = GetElementPtrInst::getTypeAtIndex(current_type, index);
    }
    kgepi->offset = offset;
  }
}

void Executor::bindModuleConstants() {
  for (auto &kfp : kmodule->functions) {
    KFunction *kf = kfp.get();
    for (unsigned i=0; i<kf->numInstructions; ++i)
      bindInstructionConstants(kf->instructions[i]);
  }

  kmodule->constantTable =
      std::unique_ptr<Cell[]>(new Cell[kmodule->constants.size()]);
  for (unsigned i=0; i<kmodule->constants.size(); ++i) {
    Cell &c = kmodule->constantTable[i];
    c.value = evalConstant(kmodule->constants[i]);
  }
}

bool Executor::checkMemoryUsage() {
  if (!MaxMemory) return true;

  // to pummel the freelist once we hit the memory cap.
  if ((stats::instructions & 0xFFFFU) != 0) // every 65536 instructions
    return true;

  // check memory limit
  const auto mallocUsage = util::GetTotalMallocUsage() >> 20U;
  const auto mmapUsage = memory->getUsedDeterministicSize() >> 20U;
  const auto totalUsage = mallocUsage + mmapUsage;
  atMemoryLimit = totalUsage > MaxMemory; // inhibit forking
  if (!atMemoryLimit)
    return true;

  if (totalUsage <= MaxMemory + 100)
    return true;

  // just guess at how many to kill
  const auto numStates = states.size();
  auto toKill = std::max(1UL, numStates - numStates * MaxMemory / totalUsage);
  klee_warning("killing %lu states (over memory cap: %luMB)", toKill, totalUsage);

  // randomly select states for early termination
  std::vector<ExecutionState *> arr(states.begin(), states.end()); // FIXME: expensive
  for (unsigned i = 0, N = arr.size(); N && i < toKill; ++i, --N) {
    unsigned idx = theRNG.getInt32() % N;
    // Make two pulls to try and not hit a state that
    // covered new code.
    if (arr[idx]->coveredNew)
      idx = theRNG.getInt32() % N;

    std::swap(arr[idx], arr[N - 1]);
    terminateStateEarly(*arr[N - 1], "Memory limit exceeded.", StateTerminationType::OutOfMemory);
  }

  return false;
}

void Executor::doDumpStates() {
  if (!DumpStatesOnHalt || states.empty()) {
    interpreterHandler->incPathsExplored(states.size());
    return;
  }

  klee_message("halting execution, dumping remaining states");
  for (const auto &state : states)
    terminateStateEarly(*state, "Execution halting.", StateTerminationType::Interrupted);
  updateStates(nullptr);
}

void Executor::run(ExecutionState &initialState) {
  bindModuleConstants();

  // Delay init till now so that ticks don't accrue during optimization and such.
  timers.reset();

  states.insert(&initialState);

  if (usingSeeds) {
    std::vector<SeedInfo> &v = seedMap[&initialState];
    
    for (std::vector<KTest*>::const_iterator it = usingSeeds->begin(), 
           ie = usingSeeds->end(); it != ie; ++it)
      v.push_back(SeedInfo(*it));

    int lastNumSeeds = usingSeeds->size()+10;
    time::Point lastTime, startTime = lastTime = time::getWallTime();
    ExecutionState *lastState = 0;
    while (!seedMap.empty()) {
      if (haltExecution) {
        doDumpStates();
        return;
      }

      std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it = 
        seedMap.upper_bound(lastState);
      if (it == seedMap.end())
        it = seedMap.begin();
      lastState = it->first;
      ExecutionState &state = *lastState;
      KInstruction *ki = state.pc;
      stepInstruction(state);

      executeInstruction(state, ki);
      if (::dumpStates) dumpStates();
      if (::dumpExecutionTree)
        dumpExecutionTree();
      updateStates(&state);

      if ((stats::instructions % 1000) == 0) {
        int numSeeds = 0, numStates = 0;
        for (std::map<ExecutionState*, std::vector<SeedInfo> >::iterator
               it = seedMap.begin(), ie = seedMap.end();
             it != ie; ++it) {
          numSeeds += it->second.size();
          numStates++;
        }
        const auto time = time::getWallTime();
        const time::Span seedTime(SeedTime);
        if (seedTime && time > startTime + seedTime) {
          klee_warning("seed time expired, %d seeds remain over %d states",
                       numSeeds, numStates);
          break;
        } else if (numSeeds<=lastNumSeeds-10 ||
                   time - lastTime >= time::seconds(10)) {
          lastTime = time;
          lastNumSeeds = numSeeds;          
          klee_message("%d seeds remaining over: %d states", 
                       numSeeds, numStates);
        }
      }
    }

    klee_message("seeding done (%d states remain)", (int) states.size());

    if (OnlySeed) {
      doDumpStates();
      return;
    }
  }

  searcher = constructUserSearcher(*this);

  std::vector<ExecutionState *> newStates(states.begin(), states.end());
  searcher->update(0, newStates, std::vector<ExecutionState *>());

  // main interpreter loop
  while (!states.empty() && !haltExecution) {
    ExecutionState &state = searcher->selectState();
    KInstruction *ki = state.pc;
    stepInstruction(state);

    executeInstruction(state, ki);
    if (::dumpStates) dumpStates();
    if (::dumpExecutionTree)
      dumpExecutionTree();
    

    updateStates(&state);

    if (!checkMemoryUsage()) {
      // update searchers when states were terminated early due to memory pressure
      updateStates(nullptr);
    }
  }

  delete searcher;
  searcher = nullptr;

  doDumpStates();
}

std::string Executor::getAddressInfo(ExecutionState &state, 
                                     ref<Expr> address) const{
  std::string Str;
  llvm::raw_string_ostream info(Str);
  info << "\taddress: " << address << "\n";
  uint64_t example;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(address)) {
    example = CE->getZExtValue();
  } else {
    ref<ConstantExpr> value;
    bool success = solver->getValue(state.constraints, address, value,
                                    state.queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    example = value->getZExtValue();
    info << "\texample: " << example << "\n";
    std::pair<ref<Expr>, ref<Expr>> res =
        solver->getRange(state.constraints, address, state.queryMetaData);
    info << "\trange: [" << res.first << ", " << res.second <<"]\n";
  }
  
  MemoryObject hack((unsigned) example);    
  MemoryMap::iterator lower = state.addressSpace.objects.upper_bound(&hack);
  info << "\tnext: ";
  if (lower==state.addressSpace.objects.end()) {
    info << "none\n";
  } else {
    const MemoryObject *mo = lower->first;
    std::string alloc_info;
    mo->getAllocInfo(alloc_info);
    info << "object at " << mo->address
         << " of size " << mo->size << "\n"
         << "\t\t" << alloc_info << "\n";
  }
  if (lower!=state.addressSpace.objects.begin()) {
    --lower;
    info << "\tprev: ";
    if (lower==state.addressSpace.objects.end()) {
      info << "none\n";
    } else {
      const MemoryObject *mo = lower->first;
      std::string alloc_info;
      mo->getAllocInfo(alloc_info);
      info << "object at " << mo->address 
           << " of size " << mo->size << "\n"
           << "\t\t" << alloc_info << "\n";
    }
  }

  return info.str();
}

void Executor::terminateState(ExecutionState &state,
                              StateTerminationType reason) {
  if (std::find(removedStates.begin(), removedStates.end(), &state) !=
      removedStates.end())
    return;

  if (replayKTest && replayPosition!=replayKTest->numObjects) {
    klee_warning_once(replayKTest,
                      "replay did not consume all objects in test input.");
  }

  interpreterHandler->incPathsExplored();
  executionTree->setTerminationType(state, reason);

  std::vector<ExecutionState *>::iterator it =
      std::find(addedStates.begin(), addedStates.end(), &state);
  if (it==addedStates.end()) {
    state.pc = state.prevPC;

    removedStates.push_back(&state);
  } else {
    // never reached searcher, just delete immediately
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(&state);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    addedStates.erase(it);
    executionTree->remove(state.executionTreeNode);
    delete &state;
  }
}

static bool shouldWriteTest(const ExecutionState &state) {
  return !OnlyOutputStatesCoveringNew || state.coveredNew;
}

static std::string terminationTypeFileExtension(StateTerminationType type) {
  std::string ret;
  #undef TTYPE
  #undef TTMARK
  #define TTYPE(N,I,S) case StateTerminationType::N: ret = (S); break;
  #define TTMARK(N,I)
  switch (type) {
    TERMINATION_TYPES
  }
  return ret;
};

void Executor::terminateStateOnExit(ExecutionState &state) {
  ++stats::terminationExit;
  //       state, nullptr,

  interpreterHandler->incPathsCompleted();
  terminateState(state, StateTerminationType::Exit);
}

void Executor::terminateStateEarly(ExecutionState &state, const Twine &message,
                                   StateTerminationType reason) {
  if (reason <= StateTerminationType::EARLY) {
    assert(reason > StateTerminationType::EXIT);
    ++stats::terminationEarly;
  }


  terminateState(state, reason);
}

void Executor::terminateStateEarlyAlgorithm(ExecutionState &state,
                                            const llvm::Twine &message,
                                            StateTerminationType reason) {
  assert(reason > StateTerminationType::EXECERR &&
         reason <= StateTerminationType::EARLYALGORITHM);
  ++stats::terminationEarlyAlgorithm;
  terminateStateEarly(state, message, reason);
}

void Executor::terminateStateEarlyUser(ExecutionState &state,
                                       const llvm::Twine &message) {
  ++stats::terminationEarlyUser;
  terminateStateEarly(state, message, StateTerminationType::SilentExit);
}

const InstructionInfo & Executor::getLastNonKleeInternalInstruction(const ExecutionState &state,
    Instruction ** lastInstruction) {
  // unroll the stack of the applications state and find
  // the last instruction which is not inside a KLEE internal function
  ExecutionState::stack_ty::const_reverse_iterator it = state.stack.rbegin(),
      itE = state.stack.rend();

  itE--;

  const InstructionInfo * ii = 0;
  if (kmodule->internalFunctions.count(it->kf->function) == 0){
    ii =  state.prevPC->info;
    *lastInstruction = state.prevPC->inst;
    //  Cannot return yet because even though
    //  been called from an internal function.
  }

  // Wind up the stack and check if we are in a KLEE internal function.
  // We visit the entire stack because we want to return a CallInstruction
  // that was not reached via any KLEE internal functions.
  for (;it != itE; ++it) {
    // check calling instruction and if it is contained in a KLEE internal function
    const Function * f = (*it->caller).inst->getParent()->getParent();
    if (kmodule->internalFunctions.count(f)){
      ii = 0;
      continue;
    }
    if (!ii){
      ii = (*it->caller).info;
      *lastInstruction = (*it->caller).inst;
    }
  }

  if (!ii) {
    // something went wrong, play safe and return the current instruction info
    *lastInstruction = state.prevPC->inst;
    return *state.prevPC->info;
  }
  return *ii;
}

bool shouldExitOn(StateTerminationType reason) {
  auto it = std::find(ExitOnErrorType.begin(), ExitOnErrorType.end(), reason);
  return it != ExitOnErrorType.end();
}

void Executor::terminateStateOnError(ExecutionState &state,
                                     const llvm::Twine &messaget,
                                     StateTerminationType terminationType,
                                     const llvm::Twine &info,
                                     const char *suffix) {
  std::string message = messaget.str();
  static std::set< std::pair<Instruction*, std::string> > emittedErrors;
  Instruction * lastInst;
  const InstructionInfo &ii = getLastNonKleeInternalInstruction(state, &lastInst);
  storeBuggyQuery(state, terminationType, message, ii.file.empty() ? -1 : ii.line, ii.assemblyLine);
  bool print_bug = true;
  if (terminationType != StateTerminationType::Ptr && terminationType != StateTerminationType::Overflow && terminationType != StateTerminationType::RACE) {
    print_bug = false;
  }

  if (terminationType == StateTerminationType::Ptr && (message.find("out of bound") == std::string::npos) || (message.find("null pointer") != std::string::npos)) {
    print_bug = false;
  }

  if (print_bug) {
    if (!ii.file.empty()) {
      klee_message("Bug Detected: %s:%d: %s", ii.file.c_str(), ii.line, message.c_str());
    } else {
      if (ii.assemblyLine)
        klee_message("Bug Detected: assemblyLine %d: %s", ii.assemblyLine, message.c_str());
      else
        klee_message("Bug Detected: (location information missing) %s", message.c_str());
    }

    std::string MsgString;
    llvm::raw_string_ostream msg(MsgString);
    msg << "Bug Detected: " << message << '\n';
    if (!ii.file.empty()) {
      msg << "File: " << ii.file << '\n'
          << "Line: " << ii.line << '\n'
          << "assembly.ll line: " << ii.assemblyLine << '\n'
          << "State: " << state.getID() << '\n';
    }
    msg << "Stack: \n";
    state.dumpStack(msg);

    std::string info_str = info.str();
    if (!info_str.empty())
      msg << "Info: \n" << info_str;

  }

  terminateState(state, terminationType);

  if (shouldExitOn(terminationType))
    haltExecution = true;
}

void Executor::storeBuggyQuery(ExecutionState &state, StateTerminationType reason, std::string message, unsigned sourceLine, unsigned assemblyLine) {
  if (isLoopCheck) return;
  if (reason != StateTerminationType::Ptr && reason != StateTerminationType::Overflow && reason != StateTerminationType::RACE) {
    return;
  }
  if (reason == StateTerminationType::Ptr && (message.find("out of bound") == std::string::npos) || (message.find("null pointer") != std::string::npos)) {
    return;
  }

  std::string bugType = "";
  if (reason == StateTerminationType::Ptr) {
    bugType="oob";
  } else if (reason == StateTerminationType::Overflow) {
    bugType="io";
  } else if (reason == StateTerminationType::RACE) {
    bugType="dr";
  }
  
  unsigned callerSourceLine = 0;
  unsigned callerAsmLine = 0;
  if (!state.stack.empty()) {
    const StackFrame &lastFrame = state.stack.back();
    if (lastFrame.caller) {
      if (auto *callerki = dyn_cast<KInstruction>(&*lastFrame.caller)) {
        callerAsmLine = callerki->info->assemblyLine;
        if (sourceLine > 0)
          callerSourceLine = callerki->info->line;
      }
    }
  }

  unsigned grandCallerSourceLine = 0;
  unsigned grandCallerAsmLine = 0;
  if (state.stack.size() >= 2) {
    const StackFrame &grandCallerFrame = state.stack[state.stack.size() - 2];
    if (grandCallerFrame.caller) {
      if (auto *grandCallerKI = dyn_cast<KInstruction>(&*grandCallerFrame.caller)) {
        grandCallerAsmLine = grandCallerKI->info->assemblyLine;
        if (sourceLine > 0)
          grandCallerSourceLine = grandCallerKI->info->line;
      }
    }
  }

  std::string filename = llvm::utostr(assemblyLine);
  if (callerAsmLine) {
    filename+="_"+llvm::utostr(callerAsmLine);
  }
  if (grandCallerAsmLine) {
    filename+="_"+llvm::utostr(grandCallerAsmLine);
  }
  filename = "asm-"+filename;

  std::string tmp = "";
  if (sourceLine > 0) {
    tmp = llvm::utostr(sourceLine) ;
  }
  if (callerSourceLine > 0) {
    tmp+="_"+llvm::utostr(callerSourceLine);
  }
  if (grandCallerSourceLine > 0) {
    tmp+="_"+llvm::utostr(grandCallerSourceLine);
  }
  if (!tmp.empty()) {
    filename = tmp+"-"+filename;
  }

  filename+="_"+bugType;

  std::string queryStr;
  getConstraintLog(state, queryStr, Interpreter::LogType::STP);

  std::string error;
  std::unique_ptr<llvm::raw_fd_ostream> dumpedQueriesFile = klee_open_output_file(outputDir+"/"+filename+".txt", error);
  if (!dumpedQueriesFile) {
    return;
  }
  
  *dumpedQueriesFile << "; start Z3 query\n";
  *dumpedQueriesFile << queryStr;
  *dumpedQueriesFile << "(get-model)\n";
  *dumpedQueriesFile << "(reset)\n";
  *dumpedQueriesFile << "; end Z3 query\n\n";

  
  dumpedQueriesFile->flush();
}

void Executor::terminateStateOnExecError(ExecutionState &state,
                                         const llvm::Twine &message,
                                         StateTerminationType reason) {
  assert(reason > StateTerminationType::USERERR &&
         reason <= StateTerminationType::EXECERR);
  ++stats::terminationExecutionError;
  terminateStateOnError(state, message, reason, "");
}

void Executor::terminateStateOnProgramError(ExecutionState &state,
                                            const llvm::Twine &message,
                                            StateTerminationType reason,
                                            const llvm::Twine &info,
                                            const char *suffix) {
  assert(reason > StateTerminationType::SOLVERERR &&
         reason <= StateTerminationType::PROGERR);
  ++stats::terminationProgramError;
  terminateStateOnError(state, message, reason, info, suffix);
}

void Executor::terminateStateOnSolverError(ExecutionState &state,
                                           const llvm::Twine &message) {
  ++stats::terminationSolverError;
  terminateStateOnError(state, message, StateTerminationType::Solver, "");
}

void Executor::terminateStateOnUserError(ExecutionState &state,
                                         const llvm::Twine &message,
                                         bool writeErr) {
  ++stats::terminationUserError;
  if (writeErr) {
    terminateStateOnError(state, message, StateTerminationType::User, "");
  } else {
    terminateState(state, StateTerminationType::User);
  }
}

// XXX shoot me
static const char *okExternalsList[] = { "printf", 
                                         "fprintf", 
                                         "puts",
                                         "getpid" };
static std::set<std::string> okExternals(okExternalsList,
                                         okExternalsList + 
                                         (sizeof(okExternalsList)/sizeof(okExternalsList[0])));

unsigned Executor::getTypeBits(llvm::Type *returnType) {
  unsigned width = 0;
  if (returnType->isIntegerTy()) {
    width = returnType->getIntegerBitWidth();
  } else if (returnType->isFloatingPointTy()) {
    if (returnType->isHalfTy()) width = 16;
    else if (returnType->isFloatTy()) width = 32;
    else if (returnType->isDoubleTy()) width = 64;
    else if (returnType->isFP128Ty() || returnType->isPPC_FP128Ty()) width = 128;
  } else if (returnType->isPointerTy()) {
    width = Context::get().getPointerWidth();
  }
  return width;
}

// https://github.com/pytorch/pytorch/blob/main/c10/core/ScalarType.h
unsigned Executor::getTensorTypeBytes(ref<Expr> scalarType) {
  unsigned size = 0;
  if (auto typeCE = dyn_cast<ConstantExpr>(scalarType)) {
    uint typeVal = typeCE->getZExtValue();
    if (typeVal == 0 || typeVal == 1) {
      size = 1;
    } else if (typeVal == 2) {
      size = 2;
    } else if (typeVal == 3) {
      size = 4;
    } else if (typeVal == 4) {
      size = 8;
    } else if (typeVal == 5) {
      size = 2;
    } else if (typeVal == 6) {
      size = 4;
    } else if (typeVal == 7) {
      size = 8;
    } else if (typeVal == 8) {
      size = 4;
    } else if (typeVal == 9) {
      size = 8;
    } else if (typeVal == 10) {
      size = 16;
    } else if (typeVal == 11 || typeVal == 12 || typeVal == 13) {
      size = 1;
    } else if (typeVal == 14) {
      size = 4;
    } else if (typeVal == 15) {
      size = 2;
    } else if (typeVal >= 16 && typeVal <= 21) {
      size = 1;
    } else if (typeVal == 22) {
      size = 2;
    }  else if (typeVal >= 23 && typeVal <= 26) {
      size = 1;
    }else if (typeVal == 27) {
      size = 2;
    } else if (typeVal == 28) {
      size = 4;
    } else if (typeVal == 29) {
      size = 8;
    }    
  }
  return size;
}

ref<Expr> Executor::getScalarTypeByStr(std::string typeStr) {
  std::string dtypes[30] = {"torch.uint8", "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.float16", "torch.float32", "torch.float64", "torch.complex32", "torch.complex64", "torch.complex128",
    "torch.bool", "torch.qint8", "torch.quint8", "torch.qint32", "torch.bfloat16",
    "torch.quint4x2", "torch.quint2x4", "torch.bits1x8", "torch.bits2x4", "torch.bits4x2", "torch.bits8", "torch.bits16", 
    "torch.float8_e5m2", "torch.float8_e4m3fn", "torch.float8_e5m2fnuz", "torch.float8_e4m3fnuz",
    "torch.uint16", "torch.uint32", "torch.uint64"};
  
  for (int i = 0; i < 30; i++) {
    if (dtypes[i] == typeStr) {
      return ConstantExpr::create(i, Expr::Int8);
    }
  }
  return nullptr;
}

ref<Expr> Executor::getAccumalateTensorType(ref<Expr> scalarType) {
  unsigned acutype = 0;
  if (auto typeCE = dyn_cast<ConstantExpr>(scalarType)) {
    uint typeVal = typeCE->getZExtValue();
    if (typeVal == 5 || typeVal == 15 || (typeVal >= 23 && typeVal <= 26)) {
      acutype = 6;
    } else if (typeVal <= 4 || (typeVal >= 12 && typeVal <= 14)) {
      acutype = 4;
    } else if (typeVal == 8) {
      acutype = 9;
    } else {
      acutype = typeVal;
    }    
    return ConstantExpr::create(acutype, scalarType->getWidth());
  }
  return scalarType;
}

void Executor::setElementSize(ExecutionState &state, SymArrayMemoryObject *symmo) {
  if (!symmo->elementSize.isNull() && isa<ConstantExpr>(symmo->elementSize)) {
    return;
  }

  if (!symmo->scalarType.isNull()) {
    ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(symmo->scalarType);
    if (!readExpr.isNull()) {
      std::string readName = readExpr->updates.root->name;
      std::string suffix = ".scalar_type";
      if (readName.size() >= suffix.size()) {
        if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
          std::string tensorName = readName.substr(0, readName.size() - suffix.size());
          if (tensorName != symmo->arrayName->name) {
            SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[tensorName];
            if (sourceSymmo->scalarType && isa<ConstantExpr>(sourceSymmo->scalarType)) {
              symmo->scalarType = sourceSymmo->scalarType;
            }
            if (sourceSymmo->elementSize && isa<ConstantExpr>(sourceSymmo->elementSize)) {
              symmo->elementSize = sourceSymmo->elementSize;
            }
          }
        }
      }
    }
  }

  if (!symmo->elementSize.isNull() && isa<ConstantExpr>(symmo->elementSize)) {
    return;
  }

  if (!symmo->scalarType.isNull() && isa<ConstantExpr>(symmo->scalarType)) {
    unsigned size = getTensorTypeBytes(symmo->scalarType);
    if (size == 0) {
      klee_warning("tensor.element_size: not handled type");
      std::string sizeName = symmo->arrayName->name + ".item_size";
      MemoryObject *sizeMo =
        memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                        &state, /*allocSite=*/state.prevPC->inst,
                        /*alignment=*/8);          
      ref<Expr> sizeExpr = symbolizeIndex(state, sizeName, sizeMo);
      symmo->elementSize = sizeExpr;
      addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
      addConstraint(state, UleExpr::create(sizeExpr, ConstantExpr::create(16, sizeExpr->getWidth())));
    } else
      symmo->elementSize = ConstantExpr::create(size, Expr::Int64);
  }
}

void Executor::computeTensorIteratorShape(ExecutionState &state) {
  if (!state.inputIterator->shape.empty()) return;

  int index = 0;
  for (auto name: state.inputIterator->tensors) {
    if (state.inputIterator->noutputs > 0 && index < state.inputIterator->noutputs && state.inputIterator->resizeOutput) continue;
    SymArrayMemoryObject *symmo = state.symbolicArrayMap[name];
    if (symmo->shapeSize.isNull() || !isa<ConstantExpr>(symmo->shapeSize)) {
      klee_error("TensorIterator: not support symbolic dim");
    }
    auto shapeSizeCE = dyn_cast<ConstantExpr>(symmo->shapeSize);
    int shapeSize = shapeSizeCE->getZExtValue();

    if (state.inputIterator->shape.empty()) {
      for (int i = 0; i < shapeSize; i++){
        ref<Expr> dimSize = symmo->dimensionSize[i];
        state.inputIterator->shape.push_back(dimSize);
      }
    } else {
      int dimsA = state.inputIterator->shape.size();
      int maxShapeSize = shapeSize > dimsA ? shapeSize : dimsA;
      state.inputIterator->shape.resize(maxShapeSize); 
      for (int i = maxShapeSize - 1; i >= 0; i--){
        int offset = maxShapeSize - 1 - i;
        int dimA = dimsA - 1 - offset;
        int dimB = shapeSize - 1 - offset;
        
        if (dimA >= 0 && dimB >= 0) {
          ref<Expr> sizeA = state.inputIterator->shape[dimA];
          ref<Expr> sizeB = symmo->dimensionSize[dimB];
          addConstraint(state, OrExpr::create(OrExpr::create(EqExpr::create(sizeA, ZExtExpr::create(sizeB, sizeA->getWidth())), EqExpr::create(sizeB, ConstantExpr::create(1, sizeB->getWidth()))), EqExpr::create(sizeA, ConstantExpr::create(1, sizeA->getWidth()))));
          ref<Expr> finalDimSize = SelectExpr::create(EqExpr::create(sizeA, ConstantExpr::create(1, sizeA->getWidth())), ZExtExpr::create(sizeB, sizeA->getWidth()), sizeA);
          state.inputIterator->shape[i] = finalDimSize;
        } else if (dimA >= 0) {
          state.inputIterator->shape[i] = state.inputIterator->shape[dimA];;
        } else if (dimB >= 0) {
          state.inputIterator->shape[i] = symmo->dimensionSize[dimB];
        }
      }
    }
    index+=1;
  }
}

void Executor::computeTensorIteratorStrides(ExecutionState &state, SymArrayMemoryObject *symmo) {
  computeTensorIteratorShape(state);
  int ndim = state.inputIterator->shape.size();

  if (!symmo->strides_bytes.empty() || ndim == 0) return;
  if (symmo->shapeSize.isNull() || !isa<ConstantExpr>(symmo->shapeSize)) {
    klee_error("TensorIterator: not support symbolic dim");
  }

  auto shapeSizeCE = dyn_cast<ConstantExpr>(symmo->shapeSize);
  int shapeSize = shapeSizeCE->getZExtValue();
  ref<Expr> elementSize = symmo->elementSize;
  int offset = ndim - shapeSize;
  if (offset < 0) return;
  symmo->strides_bytes.reserve(ndim);

  if (symmo->strides.empty()) {
    ref<Expr> strideExpr =  ConstantExpr::create(1, Expr::Int64, false);
    for (int i = shapeSize - 1; i >= 0; i--) {
      if (symmo->dimensionSize.find(i) == symmo->dimensionSize.end()) {
        std::string sizeName = symmo->arrayName->name + ".size[" + std::to_string(i)+"]";
        auto sizepair = createSizeSymbol(state, sizeName);
        ref<Expr> dimSize = sizepair.second; 
        addConstraint(state, UgeExpr::create(dimSize, ConstantExpr::create(1, dimSize->getWidth())));
        symmo->dimensionSize[i] = dimSize;
      }
      symmo->strides[i] = strideExpr;
      strideExpr = MulExpr::create(strideExpr, symmo->dimensionSize[i]);
    }
  }

  for (int i = 0; i < shapeSize; i++) {
    ref<Expr> stride0 = ConstantExpr::create(0, Expr::Int64);
    ref<Expr> stride1 = MulExpr::create(ZExtExpr::create(symmo->strides[i], Expr::Int64), ZExtExpr::create(elementSize, Expr::Int64));
    ref<Expr> cond;
    if (state.inputIterator->shape.empty() || offset + i >= ndim) {
      cond = EqExpr::create(symmo->dimensionSize[i], ConstantExpr::create(1, symmo->dimensionSize[i]->getWidth()));
    } else {
      cond = AndExpr::create(EqExpr::create(symmo->dimensionSize[i], ConstantExpr::create(1, symmo->dimensionSize[i]->getWidth())), NeExpr::create(state.inputIterator->shape[offset + i], ConstantExpr::create(1, state.inputIterator->shape[offset + i]->getWidth())));
    }
    symmo->strides_bytes[offset + i] = SelectExpr::create(cond, stride0, stride1);
  }
}

SymArrayMemoryObject* Executor::createSymArray(ExecutionState &state, ref<Expr> address, const MemoryObject *destMO, const std::string& funcName, const std::string& tensorNamePrefix) {
  SymArrayMemoryObject *destSymmo;
  ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(address);
  uint64_t destAddress = destAddressCE->getZExtValue();
  if (destAddress != destMO->address) {
    if (state.symNames.find(destAddress)!=state.symNames.end()) {
      std::string destMName = state.symNames[destAddress];
      if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
        klee_error("%s: do not find target tensor object", funcName.c_str());
      }
      destSymmo = state.symbolicArrayMap[destMName];
    } else {
      unsigned id = 0;
      std::string uniqueName = tensorNamePrefix + "0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = tensorNamePrefix + llvm::utostr(++id);
      }
      const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
      destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
      state.symbolicArrayMap[uniqueName] = destSymmo;
      state.symNames[destAddress] = uniqueName;
      state.symAddressMap[destMO->address] = destAddress;
    }
  } else if (destMO->name.empty() || state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
    unsigned id = 0;
    std::string uniqueName = tensorNamePrefix + "0";
    while (!state.arrayNames.insert(uniqueName).second) {
      uniqueName = tensorNamePrefix + llvm::utostr(++id);
    }
    destMO->setName(uniqueName);
    const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
    bindObjectInState(state, destMO, false, array);
    state.addSymbolic(destMO, array);
    destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
    state.symbolicArrayMap[destMO->name] = destSymmo;
  } else {
    destSymmo = state.symbolicArrayMap[destMO->name];
  }
  return destSymmo;
}

ref<Expr> Executor::convertFP2Int(ExecutionState &state, ref<Expr> arg, bool isSigned, Expr::Width resultType) {
  if (isa<ConstantExpr>(arg)) {
      ref<ConstantExpr> arg_constant = toConstant(state, arg,
                                       "floating point");
      llvm::APFloat Arg(*fpWidthToSemantics(arg_constant->getWidth()), arg_constant->getAPValue());

      uint64_t value = 0;
      bool isExact = true;
#if LLVM_VERSION_CODE >= LLVM_VERSION(16, 0)
      auto valueRef = llvm::MutableArrayRef(value);
#else
      auto valueRef = makeMutableArrayRef(value);
#endif
      Arg.convertToInteger(valueRef, resultType, true,
                          llvm::APFloat::rmTowardZero, &isExact);
      return ConstantExpr::alloc(value, resultType, isSigned);
    }
  if (isSigned) {
    return SExtExpr::create(arg, resultType);
  }
  return ZExtExpr::create(arg, resultType);
}

void Executor::callExternalFunction(ExecutionState &state, KInstruction *target,
                                    KCallable *callable,
                                    std::vector<ref<Expr>> &arguments) {
  // check if specialFunctionHandler wants it
  if (const auto *func = dyn_cast<KFunction>(callable);
      func &&
      specialFunctionHandler->handle(state, func->function, target, arguments))
    return;

  StringRef externalName = callable->getName();
  auto hasUnsafeHostString = [&](ref<Expr> stringExpr) {
    ObjectPair stringObject;
    bool success = false;
    if (!state.addressSpace.resolveOne(state, solver.get(), stringExpr,
                                       stringObject, success) ||
        !success || stringObject.first->size < 16)
      return true;

    ref<Expr> dataPtr = stringObject.second->read(0, Expr::Int64);
    auto dataPtrCE = dyn_cast<ConstantExpr>(dataPtr);
    if (!dataPtrCE)
      return true;

    uint64_t dataPtrValue = dataPtrCE->getZExtValue();
    if (dataPtrValue == 0xababababababababULL)
      return true;

    ref<Expr> length = stringObject.second->read(8, Expr::Int64);
    auto lengthCE = dyn_cast<ConstantExpr>(length);
    if (!lengthCE)
      return true;

    uint64_t lengthValue = lengthCE->getZExtValue();
    if (lengthValue > (1ULL << 20))
      return true;

    const uint64_t localData = stringObject.first->address + 16;
    if (dataPtrValue != localData && stringObject.first->size >= 24) {
      ref<Expr> capacity = stringObject.second->read(16, Expr::Int64);
      auto capacityCE = dyn_cast<ConstantExpr>(capacity);
      if (!capacityCE)
        return true;
      uint64_t capacityValue = capacityCE->getZExtValue();
      if (capacityValue < lengthValue || capacityValue > (1ULL << 20))
        return true;
    }

    return false;
  };

  if (externalName == "__cxa_atexit" || externalName == "atexit" ||
      externalName == "__cxa_thread_atexit_impl" ||
      externalName == "__cxa_finalize") {
    // klee_warning_once(callable->getValue(),
    //                   "ignoring external destructor registration: %s",
    //                   externalName.str().c_str());
    Type *resultType = target->inst->getType();
    if (!resultType->isVoidTy())
      bindLocal(target, state,
                ConstantExpr::create(0, getWidthForLLVMType(resultType)));
    return;
  }

  const bool isStdBasicString =
      externalName.contains("St7__cxx1112basic_string") &&
      externalName.contains("IcSt11char_traitsIcESaIcEE");
  if (isStdBasicString &&
      (externalName.contains("D0Ev") || externalName.contains("D1Ev") ||
       externalName.contains("D2Ev"))) {
    ObjectPair stringObject;
    bool success = false;
    if (!arguments.empty() &&
        state.addressSpace.resolveOne(state, solver.get(), arguments[0],
                                      stringObject, success) &&
        success && stringObject.first->size >= 8) {
      ref<Expr> dataPtr = stringObject.second->read(0, Expr::Int64);
      if (auto dataPtrCE = dyn_cast<ConstantExpr>(dataPtr)) {
        if (dataPtrCE->getZExtValue() == 0xababababababababULL)
          return;
      }
    }
  }

  const bool isRuntimeErrorFromString =
      externalName.contains("St13runtime_errorC") &&
      externalName.contains("ERKNSt7__cxx1112basic_string");
  if (isRuntimeErrorFromString && arguments.size() >= 2 &&
      hasUnsafeHostString(arguments[1])) {
    return;
  }

  if (ExternalCalls == ExternalCallPolicy::None &&
      !okExternals.count(callable->getName().str())) {
    klee_warning("Disallowed call to external function: %s\n",
                 callable->getName().str().c_str());
    terminateStateOnUserError(state, "external calls disallowed");
    return;
  }

  if (callable->getName().find("__asm__") == 0) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    unsigned width = getTypeBits(returnType);
    if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(target->inst)) {
      if (llvm::InlineAsm *IA = llvm::dyn_cast<llvm::InlineAsm>(CI->getCalledOperand())) {
        if (IA->getAsmString().find("mov.u32 $0, WARP_SZ") != std::string::npos) {
          bindLocal(target, state, ConstantExpr::create(32, Expr::Int32));
          return;
        } else if (IA->getAsmString().find("mov.u32 $0, %laneid") != std::string::npos || IA->getAsmString().find("mov.s32 $0, %laneid") != std::string::npos) {
          ref<Expr> tidXExpr = state.kernelConfig.getThreadIdx().x;
          bindLocal(target, state, URemExpr::create(tidXExpr, ConstantExpr::create(32, tidXExpr->getWidth())));
          return;
        } else if (IA->getAsmString().find("shfl.sync.bfly.b32") != std::string::npos) {
          bindLocal(target, state, arguments[0]);
          return;
        } else if (IA->getAsmString().find("shfl.sync") != std::string::npos && arguments.size() >= 4) {
          bindLocal(target, state, AddExpr::create(arguments[0], arguments[3]));
          return;
        } else if (IA->getAsmString().find("cp.async.commit_group") != std::string::npos) {
          return;
        } else if (IA->getAsmString().find("setp.ne.b32 p") != std::string::npos && IA->getAsmString().find("cp.async.cg.shared.global") != std::string::npos) { 
          if (auto CE = dyn_cast<ConstantExpr>(arguments[0])) {
            int value = CE->getZExtValue();
            if (value == 0) {
              return;
            }
          }

          unsigned bytes;
          if (auto bytesCE = dyn_cast<ConstantExpr>(arguments[3])) {
            bytes = bytesCE->getZExtValue();
          } else {
            klee_warning("cp.async.cg.shared.global bytes are not constant");
            return;
          }
          


          executeMemoryOperation(state, false, arguments[2], 0, target, bytes);
          ref<Expr> loadedVal = getDestCell(state, target).value;

          if (!loadedVal.isNull()) {
            executeMemoryOperation(state, true, arguments[1], loadedVal, target);
            getDestCell(state, target).value = 0;
          }
          return;
        } else if (IA->getAsmString().find("cp.async.cg.shared.global") != std::string::npos && arguments.size() == 3) { 
          unsigned bytes;
          if (auto bytesCE = dyn_cast<ConstantExpr>(arguments[2])) {
            bytes = bytesCE->getZExtValue();
          } else {
            klee_warning("cp.async.cg.shared.global bytes are not constant");
            return;
          }

          executeMemoryOperation(state, false, arguments[1], 0, target, bytes);
          ref<Expr> loadedVal = getDestCell(state, target).value;

          




        


          if (!loadedVal.isNull()) {
            executeMemoryOperation(state, true, arguments[0], loadedVal, target);
            getDestCell(state, target).value = 0;





          

          }
          return;
        } else if (IA->getAsmString().find("cp.async.cg.shared.global") != std::string::npos && arguments.size() == 4) { 
          if (auto CE = dyn_cast<ConstantExpr>(arguments[3])) {
            int value = CE->getZExtValue();
            if (value == 0) {
              return;
            }
          }
          unsigned bytes;
          if (auto bytesCE = dyn_cast<ConstantExpr>(arguments[2])) {
            bytes = bytesCE->getZExtValue();
          } else {
            klee_warning("cp.async.cg.shared.global bytes are not constant");
            return;
          }

          executeMemoryOperation(state, false, arguments[1], 0, target, bytes);
          ref<Expr> loadedVal = getDestCell(state, target).value;
          if (!loadedVal.isNull()) {
            executeMemoryOperation(state, true, arguments[0], loadedVal, target);
            getDestCell(state, target).value = 0;
          }
          return;
        } else if (IA->getAsmString().find("cp.async.cg.shared.global [$0], [$1], 16,") != std::string::npos && arguments.size() == 2) { 
          executeMemoryOperation(state, false, arguments[1], 0, target, 16);
          ref<Expr> loadedVal = getDestCell(state, target).value;
          if (!loadedVal.isNull()) {
            executeMemoryOperation(state, true, arguments[0], loadedVal, target);
            getDestCell(state, target).value = 0;
          }
          return;
        } else if (IA->getAsmString().find("cp.async.wait_group") != std::string::npos) {
          return;
        } else if (IA->getAsmString().find("cvt.rn.f16x2.f32") != std::string::npos) {
          bindLocal(target, state, AddExpr::create(arguments[0], arguments[1]));
          return;          
        } else if (IA->getAsmString().find(".reg .f16 low,high") != std::string::npos && IA->getAsmString().find("mov.b16 $0, low") != std::string::npos) {
          bindLocal(target, state, ExtractExpr::create(arguments[0], 0, 16));
          return;          
        } else if (IA->getAsmString().find(".reg .f16 low,high") != std::string::npos && IA->getAsmString().find("mov.b16 $0, high") != std::string::npos) {
          bindLocal(target, state, ExtractExpr::create(arguments[0], 16, 16));
          return;          
        } else if (IA->getAsmString().find("vsub4.s32.s32.s32") != std::string::npos) { // TODO: didn't consider overflow and underflow now
          std::vector<ref<Expr>> resultLanes;

          for (unsigned i = 0; i < 4; ++i) {
            ref<Expr> byte0 = ExtractExpr::create(arguments[0], i * 8, Expr::Int8);
            ref<Expr> byte1 = ExtractExpr::create(arguments[1], i * 8, Expr::Int8);

            ref<Expr> sub = SubExpr::create(byte0, byte1);
            resultLanes.push_back(sub);
          }
          ref<Expr> result = ConcatExpr::create4(resultLanes[3], resultLanes[2], resultLanes[1], resultLanes[0]);
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("vset4.u32.u32.eq") != std::string::npos) {
          ref<Expr> result = ConstantExpr::create(0, Expr::Int32);
          for (unsigned i = 0; i < 4; ++i) {
            ref<Expr> byte0 = ExtractExpr::create(arguments[0], i * 8, Expr::Int8);
            ref<Expr> byte1 = ExtractExpr::create(arguments[1], i * 8, Expr::Int8);

            ref<Expr> cmp = EqExpr::create(byte0, byte1);
            ref<Expr> shiftedCmp = ShlExpr::create(ZExtExpr::create(cmp, Expr::Int32), ConstantExpr::create(i * 8, Expr::Int32));
            result = OrExpr::create(result, shiftedCmp);
          }
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("dp4a.s32.s32") != std::string::npos) {
          bindLocal(target, state, AddExpr::create(MulExpr::create(arguments[0], arguments[1]), arguments[2]));
          return; 
        } else if (IA->getAsmString().find("mov.b32 c") != std::string::npos && IA->getAsmString().find("fma.rn.bf16x2") != std::string::npos) {
          bindLocal(target, state, MulExpr::create(arguments[0], arguments[1]));
          return; 
        } else if (IA->getAsmString()=="trap;") {
          terminateStateOnExecError(state, "reached \"unreachable\" instruction");
          return;
        } else if (IA->getAsmString().find("lop3.b32") != std::string::npos) {
          bindLocal(target, state, OrExpr::create(AndExpr::create(arguments[0], arguments[1]), arguments[2]));
          return;
        } else if (IA->getAsmString().find("fma.rn.f16x2") != std::string::npos) {
          bindLocal(target, state, AddExpr::create(MulExpr::create(arguments[0], arguments[1]), arguments[2]));
          return;
        } else if (IA->getAsmString().find("mma.sync.aligned.m16n8k16.row.col") != std::string::npos) {
          ref<Expr> r0 = AddExpr::create(arguments[0], arguments[6]);
          ref<Expr> r1 = AddExpr::create(arguments[1], arguments[7]);
          ref<Expr> r2 = AddExpr::create(arguments[2], arguments[8]);
          ref<Expr> r3 = AddExpr::create(arguments[3], arguments[9]);
          ref<Expr> result = ConcatExpr::create(ConcatExpr::create(ZExtExpr::create(r3, 32), ZExtExpr::create(r2, 32)), ConcatExpr::create(ZExtExpr::create(r1, 32), ZExtExpr::create(r0, 32)));
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32") != std::string::npos) {
          ref<Expr> result = ConcatExpr::create4(arguments[11], arguments[10], arguments[9], arguments[8]);
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("ld.cg.global.v4.u32") != std::string::npos || IA->getAsmString().find("ld.shared.v4.b32") != std::string::npos ||
                    IA->getAsmString().find("ldmatrix.sync.aligned.m8n8.x4.shared.b16") != std::string::npos || IA->getAsmString().find("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16") != std::string::npos) {
            executeMemoryOperation(state, false, arguments[0], 0, target);
            return;
          
        } else if (IA->getAsmString().find("ldmatrix.sync.aligned.m8n8.x2.shared.b16") != std::string::npos) {
          executeMemoryOperation(state, false, arguments[0], 0, target);
          return;
          
        } else if (IA->getAsmString().find("bfe.u32") != std::string::npos) {
          ref<Expr> one = ConstantExpr::create(1, Expr::Int32);
          ref<Expr> shiftedOne = ShlExpr::create(one, arguments[2]);
          ref<Expr> mask = SubExpr::create(shiftedOne, ConstantExpr::create(1, Expr::Int32));

          // Logical shift right the source by offset.
          ref<Expr> shiftedSrc = LShrExpr::create(arguments[0], arguments[1]);
          // Finally, mask the shifted value.
          ref<Expr> result = AndExpr::create(shiftedSrc, mask);

          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("bfi.u32") != std::string::npos) {
          ref<Expr> one = ConstantExpr::create(1, Expr::Int32);
          ref<Expr> mask = SubExpr::create(ShlExpr::create(one, arguments[3]), ConstantExpr::create(1, Expr::Int32));
          ref<Expr> cleared = AndExpr::create(arguments[0], NotExpr::create(ShlExpr::create(mask, arguments[2])));
          ref<Expr> inserted = ShlExpr::create(AndExpr::create(arguments[1], mask), arguments[2]);
          ref<Expr> result = OrExpr::create(cleared, inserted);
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("bfe.u64") != std::string::npos) {
          ref<Expr> one = ConstantExpr::create(1, Expr::Int64);
          ref<Expr> shiftedOne = ShlExpr::create(one, arguments[2]);
          ref<Expr> mask = SubExpr::create(shiftedOne, ConstantExpr::create(1, Expr::Int64));

          // Logical shift right the source by offset.
          ref<Expr> shiftedSrc = LShrExpr::create(arguments[0], arguments[1]);
          // Finally, mask the shifted value.
          ref<Expr> result = AndExpr::create(shiftedSrc, mask);
          
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("bfi.b64") != std::string::npos) {
          // Assume dest and src are already 64 bits.
          ref<Expr> dest = arguments[0];
          ref<Expr> src  = arguments[1];
          // Extend offset and width from 32 bits to 64 bits.
          ref<Expr> offset = ZExtExpr::create(arguments[2], Expr::Int64);
          ref<Expr> width  = ZExtExpr::create(arguments[3], Expr::Int64);

          // Create a 64-bit constant '1'.
          ref<Expr> one = ConstantExpr::create(1, Expr::Int64);
          ref<Expr> mask = SubExpr::create(ShlExpr::create(one, width),
                                          ConstantExpr::create(1, Expr::Int64));

          // Clear the destination bitfield: clear the bits where the new field will go.
          ref<Expr> cleared = AndExpr::create(dest,
                              NotExpr::create(ShlExpr::create(mask, offset)));
          // Mask the source so that only the lower 'width' bits are used, and shift them.
          ref<Expr> inserted = ShlExpr::create(AndExpr::create(src, mask), offset);

          // Combine the cleared destination with the inserted field.
          ref<Expr> result = OrExpr::create(cleared, inserted);
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("ld.global.nc.b16") != std::string::npos) {
          executeMemoryOperation(state, false, arguments[0], 0, target);
          return;

        } else if (IA->getAsmString().find("ld.global.nc.v2.u64") != std::string::npos) {
            executeMemoryOperation(state, false, arguments[0], 0, target);
            return;

        } else if (IA->getAsmString().find("ld.global.cs.v2.b32") != std::string::npos) {
          executeMemoryOperation(state, false, arguments[0], 0, target);
          return;
        } else if (IA->getAsmString().find("st.global.cs.v2.b32") != std::string::npos) {
          ref<Expr> writeVal = ConcatExpr::create(arguments[2], arguments[1]);
          executeMemoryOperation(state, true, arguments[0], writeVal, target);
          return;
        } else if (IA->getAsmString().find("st.global.cs.b32") != std::string::npos) {
          executeMemoryOperation(state, true, arguments[0], arguments[1], target);
          return;
        } else if (IA->getAsmString().find("st.global.cs.v2.u64") != std::string::npos) {
          ref<Expr> writeVal = ConcatExpr::create(arguments[2], arguments[1]);
          executeMemoryOperation(state, true, arguments[0], writeVal, target);
          return;


        } else if (IA->getAsmString().find("ld.volatile.global.u32") != std::string::npos) {
          executeMemoryOperation(state, false, arguments[0], 0, target);
          return;

        } else if (IA->getAsmString().find(".reg .b16") != std::string::npos && IA->getAsmString().find("mov.b16") != std::string::npos && IA->getAsmString().find("fma.rn.bf16") != std::string::npos) {
          bindLocal(target, state, AddExpr::create(arguments[0], arguments[1]));
          return;
        } else if (IA->getAsmString().find("cvt.rn.bf16x2.f32") != std::string::npos) {
          ref<Expr> bf16_0 = ExtractExpr::create(arguments[0], 16, 16); // arg0: bits 31:16
          ref<Expr> bf16_1 = ExtractExpr::create(arguments[1], 16, 16); // arg1: bits 31:16
          // Concatenate to make a 32-bit result: bf16_0 | bf16_1
          ref<Expr> result = ConcatExpr::create(bf16_0, bf16_1);
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("atom.add.noftz.f16x2") != std::string::npos) {
          bool success;
          ObjectPair op;
          if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
            terminateStateOnProgramError(state, "atom.add.noftz.f16x2: not find object", StateTerminationType::ReportError);
            return;
          }

          const ObjectState *os = op.second;
          ref<Expr> r0 = os->read(0, 32);
          ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
          wos->write(0, AddExpr::create(r0, arguments[1]));
          bindLocal(target, state, r0);
          return;
        } else if (IA->getAsmString().find("atom.add.noftz.f16") != std::string::npos || IA->getAsmString().find("atom.add.noftz.bf16") != std::string::npos) {
          bool success;
          ObjectPair op;
          if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
            terminateStateOnProgramError(state, "atom.add.noftz.f16: not find object", StateTerminationType::ReportError);
            return;
          }
          const ObjectState *os = op.second;
          ref<Expr> r0 = os->read(0, 16);
          ObjectState *wos = state.addressSpace.getWriteable(op.first, os);
          wos->write(0, AddExpr::create(r0, arguments[1]));
          bindLocal(target, state, r0);
          return;
        } else if (IA->getAsmString().find("fma.rn.bf16x2") != std::string::npos) {
          bindLocal(target, state, AddExpr::create(MulExpr::create(arguments[0], arguments[1]), arguments[2]));
          return;
        } else if (IA->getAsmString().find("ld.global.ca.v2.b32") != std::string::npos) {
          if (auto CE = dyn_cast<ConstantExpr>(arguments[1])) {
            int value = CE->getZExtValue();
            if (value == 0) {
              bindLocal(target, state, ConstantExpr::create(0, Expr::Int64));
              return;
            }
          }
            executeMemoryOperation(state, false, arguments[0], 0, target);
            return;

        } else if (IA->getAsmString().find("mov.b16") != std::string::npos && IA->getAsmString().find("ld.global.ca.b16") != std::string::npos) {
          if (auto CE = dyn_cast<ConstantExpr>(arguments[1])) {
            int value = CE->getZExtValue();
            if (value == 0) {
              bindLocal(target, state, ConstantExpr::create(0, Expr::Int16));
              return;
            }
          }
          executeMemoryOperation(state, false, arguments[0], 0, target);
          return;
        } else if (IA->getAsmString().find("prmt.b32 $0, $2, 0x64, 0x4140;prmt.b32 $1, $2, 0x64, 0x4342") != std::string::npos) {
          ref<Expr> a = arguments[0];
          ref<ConstantExpr> b = ConstantExpr::create(0x64, Expr::Int32);
          uint32_t sel = 0x4140;
          ref<Expr> result0 = ConstantExpr::create(0, Expr::Int32);

          for (unsigned byte = 0; byte < 4; ++byte) {
            // Extract 4 bits from sel for this output byte
            unsigned control = (sel >> (byte * 4)) & 0xF;

            ref<Expr> chosen;
            if (control < 4) {
              // Select byte from "a"
              chosen = ExtractExpr::create(a, control * 8, Expr::Int8);
            } else if (control < 8) {
              // Select byte from "b"
              chosen = ExtractExpr::create(b, (control - 4) * 8, Expr::Int8);
            } else {
              // Reserved / fill with zero
              chosen = ConstantExpr::create(0, Expr::Int8);
            }

            // Place into correct output position
            ref<Expr> shifted = ZExtExpr::create(chosen, Expr::Int32);
            shifted = ShlExpr::create(shifted, ConstantExpr::create(byte * 8, Expr::Int32));
            result0 = OrExpr::create(result0, shifted);
          }

          sel = 0x4342;
          ref<Expr> result1 = ConstantExpr::create(0, Expr::Int32);

          for (unsigned byte = 0; byte < 4; ++byte) {
            // Extract 4 bits from sel for this output byte
            unsigned control = (sel >> (byte * 4)) & 0xF;

            ref<Expr> chosen;
            if (control < 4) {
              // Select byte from "a"
              chosen = ExtractExpr::create(a, control * 8, Expr::Int8);
            } else if (control < 8) {
              // Select byte from "b"
              chosen = ExtractExpr::create(b, (control - 4) * 8, Expr::Int8);
            } else {
              // Reserved / fill with zero
              chosen = ConstantExpr::create(0, Expr::Int8);
            }

            // Place into correct output position
            ref<Expr> shifted = ZExtExpr::create(chosen, Expr::Int32);
            shifted = ShlExpr::create(shifted, ConstantExpr::create(byte * 8, Expr::Int32));
            result1 = OrExpr::create(result1, shifted);
          }
          bindLocal(target, state, ConcatExpr::create(result1, result0));
          return;
        } else if (IA->getAsmString().find("prmt.b32") != std::string::npos) {
          llvm::Type *returnType = callable->getFunctionType()->getReturnType();
          int width = kmodule->targetData->getTypeSizeInBits(returnType);
          ref<Expr> kids[width/32];
          for (int i = 0; i < width/32; ++i) {
            kids[i] = arguments[0];
          }
          ref<Expr> result = ConcatExpr::createN(width/32, kids);
          bindLocal(target, state, result);
          return;
        } else if (IA->getAsmString().find("setp.ne.b32 p") != std::string::npos && IA->getAsmString().find("mov.b32") != std::string::npos && IA->getAsmString().find("ld.global.cg.v4.b32") != std::string::npos) {
          if (auto CE = dyn_cast<ConstantExpr>(arguments[1])) {
            int value = CE->getZExtValue();
            if (value == 0) {
              bindLocal(target, state, ConstantExpr::create(0, Expr::Int128));
              return;
            }
          }
            executeMemoryOperation(state, false, arguments[0], 0, target);
            return;


        } else if (IA->getAsmString().find("mov.b32") != std::string::npos && arguments.size() == 2) {
          bindLocal(target, state, ConcatExpr::create(arguments[0], arguments[1]));
          return;
        } else if (IA->getAsmString().find("mov.b32") != std::string::npos && arguments.size() == 1) {
          if (arguments[0]->getWidth() == 32) {
            ref<Expr> low16 = ExtractExpr::create(arguments[0], 0, Expr::Int16);
            ref<Expr> high16 = ExtractExpr::create(arguments[0], 16, Expr::Int16);
            bindLocal(target, state, ConcatExpr::create(high16, low16));
            return;
          } else {
            bindLocal(target, state, ZExtExpr::create(arguments[0], Expr::Int32));
            return;
          }
        } else if (IA->getAsmString().find("cvta.to.shared.u64 u64addr") != std::string::npos && IA->getAsmString().find("cvt.u32.u64") != std::string::npos) {
          bindLocal(target, state, ExtractExpr::create(arguments[0], 0, Expr::Int32));
          return;
        } else if (IA->getAsmString().find("setp.ne.b32 p") != std::string::npos && IA->getAsmString().find("st.global.v4.b32") != std::string::npos) {
          if (auto CE = dyn_cast<ConstantExpr>(arguments[1])) {
            int value = CE->getZExtValue();
            if (value == 0) {
              return;
            }
          }

          ref<Expr> writeVal = ConcatExpr::create(arguments[5], ConcatExpr::create(arguments[4], ConcatExpr::create(arguments[3], arguments[2])));
          executeMemoryOperation(state, true, arguments[0], writeVal, target);
          return;
        } else if (IA->getAsmString().find("reg .f16 a, b, c, d") != std::string::npos && IA->getAsmString().find("cvt.rn.f16.f32 a") != std::string::npos
                  && IA->getAsmString().find("cvt.rn.f16.f32 b") != std::string::npos && IA->getAsmString().find("cvt.rn.f16.f32 c") != std::string::npos
                  && IA->getAsmString().find("cvt.rn.f16.f32 a") != std::string::npos) {
          ref<Expr> arg0 = convertFP2Int(state, arguments[0], true, Expr::Int16);
          ref<Expr> arg1 = convertFP2Int(state, arguments[1], true, Expr::Int16);
          ref<Expr> arg2 = convertFP2Int(state, arguments[2], true, Expr::Int16);
          ref<Expr> arg3 = convertFP2Int(state, arguments[3], true, Expr::Int16);
          ref<Expr> res = ConcatExpr::create(ConcatExpr::create(arg3, arg2), ConcatExpr::create(arg1, arg0)); 
          bindLocal(target, state, res);
          return;
        } else if (IA->getAsmString().find(".reg .u32 start") != std::string::npos && IA->getAsmString().find(".reg .u64 extended;") != std::string::npos
                  && IA->getAsmString().find("mov.u32 start, %reserved_smem_offset_1") != std::string::npos && IA->getAsmString().find("cvt.u64.u32 extended, start") != std::string::npos
                  && IA->getAsmString().find("cvta.shared.u64 $0, extended;") != std::string::npos) {
          uint64_t shared_ptr = 0x010000ULL + 0x100ULL;
          bindLocal(target, state, ConstantExpr::create(shared_ptr, Expr::Int64));
          return;
        } else if (IA->getAsmString().find(".reg .pred __$$temp3") != std::string::npos && IA->getAsmString().find("setp.lt.f16  __$$temp3,") != std::string::npos
                  && IA->getAsmString().find("selp.u16") != std::string::npos) {
          bindLocal(target, state, SelectExpr::create(SltExpr::create(arguments[0], arguments[1]), ConstantExpr::create(1, Expr::Int16), ConstantExpr::create(0, Expr::Int16)));
          return;
        } else if (IA->getAsmString().find("prefetch.global.L2") != std::string::npos) {
          return;
        }

        if (IA->getAsmString().find("mul.") != std::string::npos && width > 0) {
          bindLocal(target, state, ZExtExpr::create(MulExpr::create(arguments[0], arguments[1]), width));
          return;
        } else if (IA->getAsmString().find("sub.") != std::string::npos && width > 0) {
          bindLocal(target, state, ZExtExpr::create(SubExpr::create(arguments[0], arguments[1]), width));
          return;
        } else if (IA->getAsmString().find("sub.") != std::string::npos && width > 0) {
          bindLocal(target, state, ZExtExpr::create(SubExpr::create(arguments[0], arguments[1]), width));
          return;
        } else if (IA->getAsmString().find("add.") != std::string::npos && width > 0) {
          bindLocal(target, state, ZExtExpr::create(AddExpr::create(arguments[0], arguments[1]), width));
          return;
        }
      }
    }

    if (arguments.size() == 1 && width > 0) {
      bindLocal(target, state, ZExtExpr::create(arguments[0], width));
      return;
    }
  }

  if (callable->getName() == "__nv_cvta_generic_to_shared_impl") {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    unsigned width = getTypeBits(returnType);
    bindLocal(target, state, ZExtExpr::create(arguments[0], width));
    return;
  }

  if (callable->getName() == "__nv_isGlobal_impl" || callable->getName() == "__nv_isShared_impl") {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    unsigned width = getTypeBits(returnType);
    bindLocal(target, state, ConstantExpr::create(1, width));
    return;
  }


  if (callable->getName().find("cudaGetLastError") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("getCurrentCUDAStream") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int128));
    return;
  }

  if (callable->getName().find("cuda10CUDAStream6streamE") != std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    bindLocal(target, state, ConstantExpr::create(0, getTypeBits(returnType)));
    return;
  }

  if (callable->getName().find("aoti_torch_get_current_stream") != std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    bindLocal(target, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
    return;
  }

  if (callable->getName().find("cublasSetStream_v2") != std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    bindLocal(target, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
    return;
  }

  if (callable->getName().find("cublasSetStream_v2") != std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    bindLocal(target, state, ConstantExpr::create(0, returnType->getIntegerBitWidth()));
    return;
  }

  if (callable->getName().find("pthread_once") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("c104cuda14ExchangeDevice") != std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("c104cuda14MaybeSetDevice") != std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName()=="__cxa_throw") {
    return;
  }

  if (callable->getName()=="_ZN6caffe28TypeMeta26error_unsupported_typemetaES0_") {
    return;
  }

  if (callable->getName().find("cuda29c10_cuda_check_implementation") != std::string::npos) {
    return;
  }

  if (callable->getName()=="cudaGetErrorString") {
    const char *errorMsg = "no error";
    size_t msgSize = strlen(errorMsg) + 1;

    MemoryObject *mo = memory->allocate(msgSize, true, false, &state, state.prevPC->inst, 8);
    ObjectState *os = bindObjectInState(state, mo, false);
    for (size_t i = 0; i < msgSize; ++i) {
        os->write8(i, errorMsg[i]);
    }
    bindLocal(target, state, mo->getBaseExpr());
    return;
  }

  if (callable->getName()=="cudaStreamIsCapturing") {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "cudaStreamIsCapturing: not find status object", StateTerminationType::ReportError);
      return;
    }
    
    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    wos->write(0, ConstantExpr::create(1, Expr::Int32));
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName()=="cudaThreadExchangeStreamCaptureMode") {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "cudaThreadExchangeStreamCaptureMode: not find status object", StateTerminationType::ReportError);
      return;
    }
    
    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    wos->write(0, ConstantExpr::create(0, Expr::Int32));
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName()=="cudaMemset" || callable->getName()=="cudaMemsetAsync") {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "cudaThreadExchangeStreamCaptureMode: not find status object", StateTerminationType::ReportError);
      return;
    }

    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    if (auto bytesCE = dyn_cast<ConstantExpr>(arguments[2])) {
      if (auto valueCE = dyn_cast<ConstantExpr>(arguments[1])) {
        int value = valueCE->getZExtValue();
        int bytes = bytesCE->getZExtValue();
        for (int i = 0; i < bytes; i++) {
          wos->write(i, ConstantExpr::create(value, Expr::Int8));
        }
      }
    }
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName()=="cudaIpcGetMemHandle") {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("ops11add__Tensor4callERNS_6TensorERKS2_")!=std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("2at4_ops15sum_IntList_out4callERKNS_6Tensor")!=std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("ops11mul__Tensor4callERNS_6TensorERKS2_")!=std::string::npos) {
    ObjectPair op0;
    bool success;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success){
      terminateStateOnProgramError(state, "ops11mul__Tensor4callERNS_6TensorERKS2_: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success){
      terminateStateOnProgramError(state, "ops11mul__Tensor4callERNS_6TensorERKS2_: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo0 = op0.first;
    std::string symName0 = getSymName(state, mo0, arguments[0]);
    const MemoryObject *mo1 = op1.first;
    std::string symName1 = getSymName(state, mo1, arguments[1]);
    if (state.symbolicArrayMap.find(symName0) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo0 = state.symbolicArrayMap[symName0];
      if (!symmo0->shapeSize.isNull() && !isa<ConstantExpr>(symmo0->shapeSize)) {
        bindLocal(target, state, arguments[0]);
      }
      if (state.symbolicArrayMap.find(symName1) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo1 = state.symbolicArrayMap[symName1];
        if (!symmo1->shapeSize.isNull() && !isa<ConstantExpr>(symmo1->shapeSize)) {
          bindLocal(target, state, arguments[0]);
        }

        unsigned id = 0;
        std::string uniqueName = "mul_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "mul_tensor_" + llvm::utostr(++id);
        }
        MemoryObject *returnMo =
              memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                              &state, /*allocSite=*/state.prevPC->inst,
                              /*alignment=*/8);          
        returnMo->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, returnMo->size);
        bindObjectInState(state, returnMo, false, array);
        state.addSymbolic(returnMo, array);
        SymArrayMemoryObject *destSymmo = new SymArrayMemoryObject(returnMo->address, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = destSymmo;
        destSymmo->elementSize = symmo0->elementSize;
        destSymmo->scalarType = symmo0->scalarType;

        auto shapeSizeExpr0 = dyn_cast<ConstantExpr>(symmo0->shapeSize);
        int shapeSize0 = shapeSizeExpr0->getZExtValue();
        auto shapeSizeExpr1 = dyn_cast<ConstantExpr>(symmo1->shapeSize);
        int shapeSize1 = shapeSizeExpr1->getZExtValue();
        int maxShapeSize = shapeSize0>shapeSize1 ? shapeSize0:shapeSize1;
        destSymmo->shapeSize = ConstantExpr::create(maxShapeSize, Expr::Int64);
        ref<Expr> sizeExpr = nullptr;
        for (int i = 1;i<=maxShapeSize;i++) {
          ref<Expr> finalDimSize;
          if (i>=shapeSize0) {
            finalDimSize = symmo1->dimensionSize[shapeSize1-i];
          } else if (i>=shapeSize1) {
            finalDimSize = symmo0->dimensionSize[shapeSize0-i];
          } else {
            ref<Expr> dim0 = symmo0->dimensionSize[shapeSize0-i];
            ref<Expr> dim1 = symmo1->dimensionSize[shapeSize1-i];
            
            addConstraint(state, OrExpr::create(OrExpr::create(EqExpr::create(dim0, ZExtExpr::create(dim1, dim0->getWidth())), EqExpr::create(dim1, ConstantExpr::create(1, dim1->getWidth()))), EqExpr::create(dim0, ConstantExpr::create(1, dim0->getWidth()))));          
            finalDimSize = SelectExpr::create(EqExpr::create(dim0, ConstantExpr::create(1, dim0->getWidth())), ZExtExpr::create(dim1, dim0->getWidth()), dim0);
          }
          if (sizeExpr.isNull()) {
            sizeExpr = finalDimSize;
          } else {
            sizeExpr = MulExpr::create(sizeExpr, finalDimSize);
          }
          destSymmo->dimensionSize[maxShapeSize-i] = finalDimSize;
        }
        destSymmo->size = sizeExpr;
        bindLocal(target, state, returnMo->getBaseExpr());
      } else {
        terminateStateOnProgramError(state, "ops11mul__Tensor4callERNS_6TensorERKS2_: do not find symbolic tensor object", StateTerminationType::ReportError);
        return;
      }
    } else {
      terminateStateOnProgramError(state, "ops11mul__Tensor4callERNS_6TensorERKS2_: do not find symbolic tensor object", StateTerminationType::ReportError);
        return;
    }
    return;
  }

  if (callable->getName().find("at4_ops10mul_Tensor4callERKNS_6TensorES4_")!=std::string::npos) {
    ObjectPair op0;
    bool success;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success){
      terminateStateOnProgramError(state, "at4_ops10mul_Tensor4callERKNS_6TensorES4_: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success){
      terminateStateOnProgramError(state, "at4_ops10mul_Tensor4callERKNS_6TensorES4_: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op2;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success){
      terminateStateOnProgramError(state, "at4_ops10mul_Tensor4callERKNS_6TensorES4_: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo0 = op1.first;
    std::string symName0 = getSymName(state, mo0, arguments[1]);
    const MemoryObject *mo1 = op2.first;
    std::string symName1 = getSymName(state, mo1, arguments[2]);
    const MemoryObject *destMo = op0.first;
    SymArrayMemoryObject *destSymmo;

    ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t destAddress = destAddressCE->getZExtValue();
    if (destAddress != destMo->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        std::string destMName = state.symNames[destAddress];
        if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "at4_ops10mul_Tensor4callERKNS_6TensorES4_: do not find target tensor object", StateTerminationType::ReportError);
        return;
        }
        destSymmo = state.symbolicArrayMap[destMName];
      } else {
        unsigned id = 0;
        std::string uniqueName = "mul_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "mul_tensor_" + llvm::utostr(++id);
        }  
        const Array *array = arrayCache.CreateArray(uniqueName, destMo->size);
        destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = destSymmo;
        state.symNames[destAddress] = uniqueName;
        state.symAddressMap[destMo->address] = destAddress;
      }
    } else if (state.symbolicArrayMap.find(destMo->name) == state.symbolicArrayMap.end()) {
      unsigned id = 0;
      std::string uniqueName = "mul_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "mul_tensor_" + llvm::utostr(++id);
      }     
      destMo->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, destMo->size);
      bindObjectInState(state, destMo, false, array);
      state.addSymbolic(destMo, array);
      destSymmo = new SymArrayMemoryObject(destMo->address, array, nullptr, nullptr);
      state.symbolicArrayMap[uniqueName] = destSymmo;
    } else {
      destSymmo = state.symbolicArrayMap[destMo->name];
    }

    if (state.symbolicArrayMap.find(symName0) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo0 = state.symbolicArrayMap[symName0];
      if (state.symbolicArrayMap.find(symName1) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *symmo1 = state.symbolicArrayMap[symName1];
        destSymmo->elementSize = symmo0->elementSize;
        destSymmo->scalarType = symmo0->scalarType;

        if (symmo0->shapeSize.isNull() || symmo0->shapeSize.isNull()) {
          klee_warning("tensor mul: operands do not have shapeSize");
          return;
        }

        if (!isa<ConstantExpr>(symmo0->shapeSize) || !isa<ConstantExpr>(symmo1->shapeSize)) {
          destSymmo->size = symmo0->size;
          destSymmo->dimensionSize = symmo0->dimensionSize;
          destSymmo->strides = symmo0->strides;
          destSymmo->shapeSize = symmo0->shapeSize;
          return;
        }

        auto shapeSizeExpr0 = dyn_cast<ConstantExpr>(symmo0->shapeSize);
        int shapeSize0 = shapeSizeExpr0->getZExtValue();
        auto shapeSizeExpr1 = dyn_cast<ConstantExpr>(symmo1->shapeSize);
        int shapeSize1 = shapeSizeExpr1->getZExtValue();
        int maxShapeSize = shapeSize0>shapeSize1 ? shapeSize0:shapeSize1;
        destSymmo->shapeSize = ConstantExpr::create(maxShapeSize, Expr::Int64);
        ref<Expr> sizeExpr = nullptr;
        for (int i = 1;i<=maxShapeSize;i++) {
          ref<Expr> finalDimSize;
          if (i>=shapeSize0) {
            finalDimSize = symmo1->dimensionSize[shapeSize1-i];
          } else if (i>=shapeSize1) {
            finalDimSize = symmo0->dimensionSize[shapeSize0-i];
          } else {
            ref<Expr> dim0 = symmo0->dimensionSize[shapeSize0-i];
            ref<Expr> dim1 = symmo1->dimensionSize[shapeSize1-i];
            
            addConstraint(state, OrExpr::create(OrExpr::create(EqExpr::create(dim0, ZExtExpr::create(dim1, dim0->getWidth())), EqExpr::create(dim1, ConstantExpr::create(1, dim1->getWidth()))), EqExpr::create(dim0, ConstantExpr::create(1, dim0->getWidth()))));          
            finalDimSize = SelectExpr::create(EqExpr::create(dim0, ConstantExpr::create(1, dim0->getWidth())), ZExtExpr::create(dim1, dim0->getWidth()), dim0);
          }
          if (sizeExpr.isNull()) {
            sizeExpr = finalDimSize;
          } else {
            sizeExpr = MulExpr::create(sizeExpr, finalDimSize);
          }
          destSymmo->dimensionSize[maxShapeSize-i] = finalDimSize;
        }
        destSymmo->size = sizeExpr;
      } else {
        terminateStateOnProgramError(state, "ops11mul__Tensor4callERNS_6TensorERKS2_: do not find symbolic tensor object", StateTerminationType::ReportError);
        return;
      }
    } else {
      terminateStateOnProgramError(state, "ops11mul__Tensor4callERNS_6TensorERKS2_: do not find symbolic tensor object", StateTerminationType::ReportError);
      return;
    }
    return;
  }

  if (callable->getName()=="_ZNK3c106SymInt6sym_neERKS0_") {
    ObjectPair op0;
    bool success;

    if (!isa<ConstantExpr>(arguments[0])) {
      auto base_it = state.base_addrs.find(arguments[0]);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, op0,
                                          success) ||
            !success) {
          terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addrs map to a memory object", StateTerminationType::ReportError);
        return;
        }
      } else {
        terminateStateOnProgramError(state, "find the base_addrs", StateTerminationType::ReportError);
        return;
      }
    } else {
      state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success);
    }
    ref<Expr> offset0 = op0.first->getOffsetExpr(arguments[0]);

    ObjectPair op1;
    if (!isa<ConstantExpr>(arguments[1])) {
      auto base_it = state.base_addrs.find(arguments[1]);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, op1,
                                          success) ||
            !success) {
          terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addrs map to a memory object", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "find the base_addrs", StateTerminationType::ReportError);
        return;
      }
    } else {
      state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success);
    }
    ref<Expr> offset1 = op1.first->getOffsetExpr(arguments[1]);
    ref<Expr> val1 = op1.second->read(offset1, Expr::Int64);

    ObjectPair op2;
    if (!isa<ConstantExpr>(arguments[2])) {
      auto base_it = state.base_addrs.find(arguments[2]);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, op2,
                                          success) ||
            !success) {
          terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addr", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "find the base_addrs", StateTerminationType::ReportError);
        return;
      }
    } else {
      state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success);
    }
    ref<Expr> offset2 = op2.first->getOffsetExpr(arguments[2]);
    ref<Expr> val2 = op2.second->read(offset2, Expr::Int64);

    ref<Expr> result = NeExpr::create(val1, val2);
    ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
    wos->write(offset0, result);
    return;
  }

  if (callable->getName()=="_ZNK3c107SymBool11expect_trueEPKcl") {
    ObjectPair op0;
    bool success;

    if (!isa<ConstantExpr>(arguments[0])) {
      auto base_it = state.base_addrs.find(arguments[0]);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, op0,
                                          success) ||
            !success) {
          terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addrs map to a memory object", StateTerminationType::ReportError);
        return;
        }
      } else {
        terminateStateOnProgramError(state, "find the base_addrs", StateTerminationType::ReportError);
        return;
      }
    } else {
      state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success);
    }
    ref<Expr> offset = op0.first->getOffsetExpr(arguments[0]);
    ref<Expr> val = op0.second->read(offset, Expr::Int8);
    bindLocal(target, state, EqExpr::create(val, ConstantExpr::create(1, Expr::Int8)));
    return;
  }

  if (callable->getName()=="_ZN3c10ltERKNS_6SymIntEi") {
    ObjectPair op0;
    bool success;

    if (!isa<ConstantExpr>(arguments[0])) {
      auto base_it = state.base_addrs.find(arguments[0]);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, op0,
                                          success) ||
            !success) {
          terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addrs map to a memory object", StateTerminationType::ReportError);
        return;
        }
      } else {
        terminateStateOnProgramError(state, "find the base_addrs", StateTerminationType::ReportError);
        return;
      }
    } else {
      state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success);
    }
    ref<Expr> offset = op0.first->getOffsetExpr(arguments[0]);
    ref<Expr> val = op0.second->read(offset, Expr::Int32);
    
    bindLocal(target, state, SltExpr::create(val, arguments[1]));
    return;
  }
  
  if (callable->getName().find("basic_string") != std::string::npos && callable->getName().find("char_traits") != std::string::npos && callable->getName().find("compare") != std::string::npos) {
    ObjectPair op0;
    bool success;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success){
      terminateStateOnProgramError(state, "string compare: do not find memory object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success){
      terminateStateOnProgramError(state, "string compare: do not find memory object", StateTerminationType::ReportError);
        return;
    }
    const ObjectState *os0 = op0.second;
    const ObjectState *os1 = op1.second;
    ref<Expr> string0 = os0->read8(8);
    ref<Expr> string1 = os1->read8(0);
    if (!isa<ConstantExpr>(string0) || !isa<ConstantExpr>(string0)) {
      bindLocal(target, state, ConstantExpr::create(0, 32));
      return;
    }
  }

  if (callable->getName()=="_ZNK2at6Tensor5indexESt16initializer_listINS_8indexing11TensorIndexEE") {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "tensor index: do not find target object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "tensor index: do not find source object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMo = op1.first;
    std::string sourceSymName = getSymName(state, sourceMo, arguments[1]);
    const MemoryObject *destMo = op0.first;
    if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
      if (auto dimSizeExpr = dyn_cast<ConstantExpr>(arguments[3])) {
        int dimSize = dimSizeExpr->getZExtValue();
        if (dimSize!=1) {
          terminateStateOnProgramError(state, "tensor index not handled", StateTerminationType::ReportError);
          return;
        }
      } else {
        terminateStateOnProgramError(state, "tensor index size not constant", StateTerminationType::ReportError);
        return;
      }
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      
      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success) {
        terminateStateOnProgramError(state, "tensor index: do not find source object", StateTerminationType::ReportError);
        return;
      }

      const ObjectState *indexArrayOS = op2.second;
      ref<Expr> indexExpr = indexArrayOS->read(0, 64);

      ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
      uint64_t destAddress = destAddressCE->getZExtValue();
      SymArrayMemoryObject *elementSymmo;
      if (destAddress != destMo->address) {
        if (state.symNames.find(destAddress)!=state.symNames.end()) {
          std::string destMName = state.symNames[destAddress];
          if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
            terminateStateOnProgramError(state, "tensor index: do not find target tensor object", StateTerminationType::ReportError);
            return;
          }
          elementSymmo = state.symbolicArrayMap[destMName];
        } else {
          std::string elementName;
          if (auto indexExprCE = dyn_cast<ConstantExpr>(indexExpr)) {
            int index = indexExprCE->getZExtValue();
            elementName = sourceSymName+"["+llvm::utostr(index)+"]";
          } else {
            unsigned id = 0;
            elementName = sourceSymName+"[symbol0]";
            while (!state.arrayNames.insert(elementName).second) {
              elementName = sourceSymName+"[symbol" + llvm::utostr(++id) +"]";
            }
          }

          ref<Expr> dim0Size;
          if (!sourceSymmo->dimensionSize.empty() && sourceSymmo->dimensionSize.find(0) != sourceSymmo->dimensionSize.end()) {
            dim0Size = sourceSymmo->dimensionSize[0];
          } else {
            std::string dim0SizeName = sourceMo->name + ".size[0]";
            auto dim0SizePair = createSizeSymbol(state, dim0SizeName);
            ref<Expr> dim0SizeExpr = dim0SizePair.second; 
            addConstraint(state, UgeExpr::create(dim0SizeExpr, ConstantExpr::create(1, dim0SizeExpr->getWidth())));  
            sourceSymmo->dimensionSize[0] = dim0SizeExpr;
            dim0Size = dim0SizeExpr;
          }
          ref<Expr> totalSize = UDivExpr::create(sourceSymmo->size, ZExtExpr::create(dim0Size, sourceSymmo->size->getWidth()));

          const Array *array = arrayCache.CreateArray(elementName, destMo->size);
          elementSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, totalSize);
          state.symNames[destAddress] = elementName;
          state.symAddressMap[destMo->address] = destAddress;
          elementSymmo->scalarType = sourceSymmo->scalarType;
          elementSymmo->elementSize = sourceSymmo->elementSize;
          elementSymmo->shapeSize = SubExpr::create(sourceSymmo->shapeSize, ConstantExpr::create(1, sourceSymmo->shapeSize->getWidth()));
          elementSymmo->maxVal = sourceSymmo->maxVal;
          elementSymmo->minVal = sourceSymmo->minVal;
          elementSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
          elementSymmo->isFloat = sourceSymmo->isFloat;
          state.symbolicArrayMap[elementName] = elementSymmo;

          if (!sourceSymmo->dimensionSize.empty()) {
            for (const auto& [key, value] : sourceSymmo->dimensionSize) {
              if (key > 0)
                elementSymmo->dimensionSize[key-1] = value;
            }
          }
        }
      } else {
        std::string elementName;
        if (auto indexExprCE = dyn_cast<ConstantExpr>(indexExpr)) {
          int index = indexExprCE->getZExtValue();
          elementName = sourceSymName+"["+llvm::utostr(index)+"]";
        } else {
          unsigned id = 0;
          elementName = sourceSymName+"[symbol0]";
          while (!state.arrayNames.insert(elementName).second) {
            elementName = sourceSymName+"[symbol" + llvm::utostr(++id) +"]";
          }
          destMo->setName(elementName);
        }
        destMo->setName(elementName);
        const Array *array = arrayCache.CreateArray(elementName, destMo->size);
        bindObjectInState(state, destMo, false, array);
        state.addSymbolic(destMo, array);

        ref<Expr> dim0Size;
        if (!sourceSymmo->dimensionSize.empty() && sourceSymmo->dimensionSize.find(0) != sourceSymmo->dimensionSize.end()) {
          dim0Size = sourceSymmo->dimensionSize[0];
        } else {
          std::string dim0SizeName = sourceMo->name + ".size[0]";
          auto dim0SizePair = createSizeSymbol(state, dim0SizeName);
          ref<Expr> dim0SizeExpr = dim0SizePair.second; 
          addConstraint(state, UgeExpr::create(dim0SizeExpr, ConstantExpr::create(1, dim0SizeExpr->getWidth())));  
          sourceSymmo->dimensionSize[0] = dim0SizeExpr;
          dim0Size = dim0SizeExpr;
        }
        ref<Expr> totalSize = UDivExpr::create(sourceSymmo->size, ZExtExpr::create(dim0Size, sourceSymmo->size->getWidth()));

        elementSymmo = new SymArrayMemoryObject(destMo->address, array, nullptr, totalSize);
        elementSymmo->scalarType = sourceSymmo->scalarType;
        elementSymmo->elementSize = sourceSymmo->elementSize;
        elementSymmo->shapeSize = SubExpr::create(sourceSymmo->shapeSize, ConstantExpr::create(1, sourceSymmo->shapeSize->getWidth()));
        elementSymmo->maxVal = sourceSymmo->maxVal;
        elementSymmo->minVal = sourceSymmo->minVal;
        elementSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
        elementSymmo->isFloat = sourceSymmo->isFloat;
        state.symbolicArrayMap[elementName] = elementSymmo;

        if (!sourceSymmo->dimensionSize.empty()) {
          for (const auto& [key, value] : sourceSymmo->dimensionSize) {
            if (key > 0)
              elementSymmo->dimensionSize[key-1] = value;
          }
        }
      }

      if (auto indexExprCE = dyn_cast<ConstantExpr>(indexExpr)) {
        int index = indexExprCE->getZExtValue();
        sourceSymmo->constantIndexAddress[index] = ConstantExpr::createPointer(elementSymmo->arrayAddress);
      } else {
        sourceSymmo->symbolicIndexAddress[arguments[2]] = ConstantExpr::createPointer(elementSymmo->arrayAddress);
      }
      return;
    } else {
      terminateStateOnProgramError(state, "tensor index: source object is not symbolic", StateTerminationType::ReportError);
        return;
    }
  }
  if (callable->getName().find("10empty_cudaEN3c108ArrayRef")!=std::string::npos || callable->getName().find("2at4_ops19empty_memory_format4callEN3c108ArrayRefINS2_6SymIntEEESt8optionalINS2_10ScalarType")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at::native:empty_cuda: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op.first;
    SymArrayMemoryObject *symmo;
    ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t destAddress = destAddressCE->getZExtValue();
    if (destAddress != mo->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        std::string destMName = state.symNames[destAddress];
        if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "at::native:empty_cuda: do not find target tensor object", StateTerminationType::ReportError);
        return;
        }
        symmo = state.symbolicArrayMap[destMName];
      } else {
        unsigned id = 0;
        std::string uniqueName = "empty_cuda_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "empty_cuda_tensor_" + llvm::utostr(++id);
        }
        const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
        symmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = symmo;
        state.symNames[destAddress] = uniqueName;
        state.symAddressMap[mo->address] = destAddress;
      }
    } else if (state.symbolicArrayMap.find(mo->name) == state.symbolicArrayMap.end()) {
      unsigned id = 0;
      std::string uniqueName = "empty_cuda_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "empty_cuda_tensor_" + llvm::utostr(++id);
      }
      mo->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
      bindObjectInState(state, mo, false, array);
      state.addSymbolic(mo, array);

      symmo = new SymArrayMemoryObject(mo->address, array, nullptr, nullptr);
      state.symbolicArrayMap[mo->name] = symmo;
    } else {
      symmo = state.symbolicArrayMap[mo->name];
    }
    
    if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
      uint64_t shapeSize = shapeSizeExpr->getZExtValue();
      ObjectPair op2;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op2, success) || !success) {
        terminateStateOnProgramError(state, "at::native:empty_cuda: do not find the shape array object", StateTerminationType::ReportError);
        return;
      }

      const ObjectState *shapeArrayOS = op2.second;
      ref<Expr> sizeExpr = nullptr;
      for (uint64_t i = 0; i < shapeSize; ++i) {
          ref<Expr> element = shapeArrayOS->read(i*8, 64);
          if (i == 0) {
            sizeExpr = element;
          } else {
            sizeExpr = MulExpr::create(sizeExpr, element);
          }
        symmo->dimensionSize[i] = element;
      }
      symmo->size = sizeExpr;
      symmo->shapeSize = arguments[2];

      ref<Expr> scalarType = ExtractExpr::create(arguments[3], 0, 8);
      symmo->scalarType = scalarType;
      unsigned size = getTensorTypeBytes(symmo->scalarType);
      if (size > 0)
        symmo->elementSize = ConstantExpr::create(size, Expr::Int64);
      else {
        bool setItemSize = false;
        ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
        if (readExpr) {
          std::string readName = readExpr->updates.root->name;
          std::string suffix = ".scalar_type";
          if (readName.size() >= suffix.size()) {
            if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
              std::string tensorName = readName.substr(0, readName.size() - suffix.size());
              if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                if (originalSymmo->elementSize) {
                  symmo->elementSize = originalSymmo->elementSize;
                  setItemSize = true;
                }
              }
            }
          }
        }
        if (!setItemSize) {
          std::string elementSizeName = mo->name + ".item_size";
          auto pair2 = createSizeSymbol(state, elementSizeName);
          ref<Expr> elementSizeExpr = pair2.second;
          symmo->elementSize = elementSizeExpr;
          addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
          addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
        }
      }
      state.symbolicArrayMap[mo->name] = symmo;
      return;
    } else {
      terminateStateOnProgramError(state, "at::native:empty_cuda: the size of the shape array is not a constant", StateTerminationType::ReportError);
        return;
    }
  }
  if (callable->getName().find("at4_ops17arange_start_step4callERKN3c106ScalarES5_S5_St8optionalINS2_10ScalarTypeEES6_INS2_6LayoutEES6_INS2_6Device")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at::ops::arange_start_step: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op.first;
    SymArrayMemoryObject *symmo;
    ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t destAddress = destAddressCE->getZExtValue();
    if (destAddress != mo->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        std::string destMName = state.symNames[destAddress];
        if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "at::ops::arange_start_step: do not find target tensor object", StateTerminationType::ReportError);
        return;
        }
        symmo = state.symbolicArrayMap[destMName];
      } else {
        unsigned id = 0;
        std::string uniqueName = "range_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "range_tensor_0" + llvm::utostr(++id);
        }
        const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
        symmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = symmo;
        state.symNames[destAddress] = uniqueName;
        state.symAddressMap[mo->address] = destAddress;
      }
    } else if (state.symbolicArrayMap.find(mo->name) == state.symbolicArrayMap.end()) {
      unsigned id = 0;
      std::string uniqueName = "range_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "range_tensor_0" + llvm::utostr(++id);
      }
      mo->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
      bindObjectInState(state, mo, false, array);
      state.addSymbolic(mo, array);

      symmo = new SymArrayMemoryObject(mo->address, array, nullptr, nullptr);
      state.symbolicArrayMap[mo->name] = symmo;
    } else {
      symmo = state.symbolicArrayMap[mo->name];
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at::ops::arange_start_step: do not find start value object", StateTerminationType::ReportError);
        return;
    }
    ref<Expr> startExpr = op1.second->read(16, 64);

    ObjectPair op2;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success) {
      terminateStateOnProgramError(state, "at::ops::arange_start_step: do not find end value object", StateTerminationType::ReportError);
        return;
    }
    ref<Expr> endExpr = op2.second->read(16, 64);

    ObjectPair op3;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[3], op3, success) || !success) {
      terminateStateOnProgramError(state, "at::ops::arange_start_step: do not find step value object", StateTerminationType::ReportError);
        return;
    }
    ref<Expr> stepExpr = op3.second->read(16, 64);
    ref<Expr> sizeExpr = UDivExpr::create(SubExpr::create(endExpr, startExpr), stepExpr);
    symmo->size = sizeExpr;

    ref<Expr> scalarType = ExtractExpr::create(arguments[4], 0, 8);
    symmo->scalarType = scalarType;
    unsigned itemSize = getTensorTypeBytes(symmo->scalarType);
    if (itemSize > 0)
      symmo->elementSize = ConstantExpr::create(itemSize, Expr::Int64);
    else {
      bool setItemSize = false;
      ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
      if (readExpr) {
        std::string readName = readExpr->updates.root->name;
        std::string suffix = ".scalar_type";
        if (readName.size() >= suffix.size()) {
          if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
            std::string tensorName = readName.substr(0, readName.size() - suffix.size());
            if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
              SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
              if (originalSymmo->elementSize) {
                symmo->elementSize = originalSymmo->elementSize;
                setItemSize = true;
              }
            }
          }
        }
      }
      if (!setItemSize) {
        std::string elementSizeName = mo->name + ".item_size";
        auto pair2 = createSizeSymbol(state, elementSizeName);
        ref<Expr> elementSizeExpr = pair2.second;
        symmo->elementSize = elementSizeExpr;
        addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
        addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
      }
    }
    
    state.symbolicArrayMap[mo->name] = symmo;
    return;
  }

  if (callable->getName().find("at6native7resize_ERKNS_6TensorEN3c108ArrayRef")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at::native::resize: do not find source tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op.first;
    if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
      uint64_t shapeSize = shapeSizeExpr->getZExtValue();
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "at::native::resize: do not find the shape array object", StateTerminationType::ReportError);
        return;
      }

      SymArrayMemoryObject *symmo;
      ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
      uint64_t destAddress = destAddressCE->getZExtValue();
      if (destAddress != mo->address) {
        if (state.symNames.find(destAddress)!=state.symNames.end()) {
          std::string destMName = state.symNames[destAddress];
          if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
            terminateStateOnProgramError(state, "at::native::resize: do not find target tensor object", StateTerminationType::ReportError);
        return;
          }
          symmo = state.symbolicArrayMap[destMName];
        } else {
          unsigned id = 0;
          std::string uniqueName = "at_resize_tensor_0";
          while (!state.arrayNames.insert(uniqueName).second) {
            uniqueName = "at_resize_tensor_" + llvm::utostr(++id);
          }
          const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
          symmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
          state.symbolicArrayMap[uniqueName] = symmo;
          state.symNames[destAddress] = uniqueName;
          state.symAddressMap[mo->address] = destAddress;
        }
      } else if (state.symbolicArrayMap.find(mo->name) == state.symbolicArrayMap.end()) {
        unsigned id = 0;
        std::string uniqueName = "at_resize_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "at_resize_tensor_" + llvm::utostr(++id);
        }
        mo->setName(uniqueName);
        const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
        bindObjectInState(state, mo, false, array);
        state.addSymbolic(mo, array);
        symmo = new SymArrayMemoryObject(mo->address, array, nullptr, nullptr);
        state.symbolicArrayMap[mo->name] = symmo;
      } else {
        symmo = state.symbolicArrayMap[mo->name];
      }

      const ObjectState *shapeArrayOS = op1.second;
      ref<Expr> sizeExpr = ConstantExpr::create(1, 64);
      for (uint64_t i = 0; i < shapeSize; ++i) {
        ref<Expr> element = shapeArrayOS->read(i*8, 64);
        symmo->dimensionSize[i] = element;
        sizeExpr = MulExpr::create(sizeExpr, element);
      }
      symmo->size = sizeExpr;
      symmo->shapeSize = arguments[2];

      setElementSize(state, symmo);
      if (!symmo->elementSize) {
        std::string elementSizeName = mo->name + ".item_size";
        auto pair2 = createSizeSymbol(state, elementSizeName);
        ref<Expr> elementSizeExpr = pair2.second;
        symmo->elementSize = elementSizeExpr;
        addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
        addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
      }
      bindLocal(target, state, arguments[0]);
      return;
    } else {
      terminateStateOnProgramError(state, "at::native::resize: the size of the shape array is not a constant", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("at6native13resize_outputERKNS_6TensorEN3c108ArrayRef")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at::native::resize_output: do not find source tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op.first;
    SymArrayMemoryObject *symmo;
    ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t destAddress = destAddressCE->getZExtValue();
    if (destAddress != mo->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        std::string destMName = state.symNames[destAddress];
        if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "at::native::resize_output: do not find target tensor object", StateTerminationType::ReportError);
        return;
        }
        symmo = state.symbolicArrayMap[destMName];
      } else {
        unsigned id = 0;
        std::string uniqueName = "at_resize_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "at_resize_tensor_" + llvm::utostr(++id);
        }
        const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
        symmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = symmo;
        state.symNames[destAddress] = uniqueName;
        state.symAddressMap[mo->address] = destAddress;
      }
    } else if (state.symbolicArrayMap.find(mo->name) == state.symbolicArrayMap.end()) {
      unsigned id = 0;
      std::string uniqueName = "at_resize_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "at_resize_tensor_" + llvm::utostr(++id);
      }
      mo->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
      bindObjectInState(state, mo, false, array);
      state.addSymbolic(mo, array);
      symmo = new SymArrayMemoryObject(mo->address, array, nullptr, nullptr);
      state.symbolicArrayMap[mo->name] = symmo;
    } else {
      symmo = state.symbolicArrayMap[mo->name];
    }

    if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
      uint64_t shapeSize = shapeSizeExpr->getZExtValue();
      ObjectPair op1;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
        terminateStateOnProgramError(state, "at::native::resize_output: do not find the shape array object", StateTerminationType::ReportError);
        return;
      }

      const ObjectState *shapeArrayOS = op1.second;
      ref<Expr> sizeExpr = ConstantExpr::create(1, 64);
      for (uint64_t i = 0; i < shapeSize; ++i) {
        ref<Expr> element = shapeArrayOS->read(i*8, 64);
        symmo->dimensionSize[i] = element;
        sizeExpr = MulExpr::create(sizeExpr, element);
      }
      symmo->size = sizeExpr;
      symmo->shapeSize = arguments[2];

      setElementSize(state, symmo);
      if (!symmo->elementSize) {
        std::string elementSizeName = mo->name + ".item_size";
        auto pair2 = createSizeSymbol(state, elementSizeName);
        ref<Expr> elementSizeExpr = pair2.second;
        symmo->elementSize = elementSizeExpr;
        addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
        addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
      }
      bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
      return;
    } else {
      std::string sizeName = symmo->arrayName->name + ".size";
      auto pair = createSizeSymbol(state, sizeName, true);
      ref<Expr> sizeExpr = pair.second;
      symmo->size = sizeExpr;
      symmo->sizeName = pair.first;
      symmo->shapeSize = arguments[2];
      setElementSize(state, symmo);
      if (!symmo->elementSize) {
        std::string elementSizeName = mo->name + ".item_size";
        auto pair2 = createSizeSymbol(state, elementSizeName);
        ref<Expr> elementSizeExpr = pair2.second;
        symmo->elementSize = elementSizeExpr;
        addConstraint(state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
        addConstraint(state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
      }
      bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
      return;
    }
  }

  if (callable->getName().find("2at4cuda24getCurrentCUDABlasHandle")!=std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    llvm::Type *elementType = returnType->getPointerElementType();

    if (auto *structType = llvm::dyn_cast<llvm::StructType>(elementType)) {
      uint64_t structSize;
      if (!structType->isSized()) {
        structSize = 64; // Assume 64 bytes
      } else {
        structSize = kmodule->targetData->getTypeAllocSize(structType);
      }
      MemoryObject *mo = memory->allocate(structSize, /*isLocal=*/true, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);
      bindLocal(target, state, mo->getBaseExpr());
    }
    return;
  }

  if (callable->getName()=="cublasHgemm") {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
    return;
  }

  if (callable->getName()=="2at4cuda4blas4gemmIN3c108BFloat16EEEvcclllNS_10OpMathTypeIT_E4typeEPKS6_lSA_lS8_PS6_l") {
    terminateStateOnProgramError(state, "2at4cuda4blas4gemmIN3c108BFloat16EEEvcclllNS_10OpMathTypeIT_E4typeEPKS6_lSA_lS8_PS6_l: not support", StateTerminationType::ReportError);
    return;
  }

  if (callable->getName().find("St7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6append")!=std::string::npos ||
      callable->getName().find("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLERKS4_")!=std::string::npos ||
      callable->getName().find("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEPKc")!=std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("cuda14current_device")!=std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    bindLocal(target, state, ConstantExpr::create(1, returnType->getIntegerBitWidth()));
    return;
  }

  if (callable->getName().find("basic_stringIcSt11char_traitsIcESaIcEED1Ev") != std::string::npos) {
    if(auto CE = dyn_cast<ConstantExpr>(arguments[0])) {
      uint64_t saddress = CE->getZExtValue();
      if (std::find(deletedAddresses.begin(), deletedAddresses.end(), saddress) != deletedAddresses.end()) {
        return;
      } else {
        deletedAddresses.push_back(saddress);
      }
    }
  }

  if (callable->getName().find("10marlin_moe") != std::string::npos && callable->getName().find("call_marlin_moe_kernel_ku") != std::string::npos && callable->getName().find("vllm10ScalarTypeEiibiiiiP11CUstream_stPK4int4S8_PS6_PKi")!= std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
    return;
  }

  if (callable->getName().find("2at38globalDeprecatedTypePropertiesRegistry") != std::string::npos) {
    MemoryObject *mo =
          memory->allocate(8, /*isLocal=*/true, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);  
    bindLocal(target, state, mo->getBaseExpr());
    return;
  }
  
  if (callable->getName().find("at32DeprecatedTypePropertiesRegistry27getDeprecatedTypeProperties") != std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at32DeprecatedTypePropertiesRegistry27getDeprecatedTypeProperties: do not find the object", StateTerminationType::ReportError);
      return;
    }
    
    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    wos->write(0, arguments[1]);
    wos->write(4, arguments[2]);
    return;
  }

  if (callable->getName().find("cudaGetDeviceProperties_v2") != std::string::npos &&
      callable->getFunctionType()->getReturnType()->isIntegerTy(32)) {
    bool success;
    ObjectPair op;
    if (arguments.empty() || !state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "cudaGetDeviceProperties_v2: do not find target object", StateTerminationType::ReportError);
      return;
    }

    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);

    for (int i = 0; i < 256; i++) {
      wos->write(i, ConstantExpr::create(0, Expr::Int8));
    }

    wos->write(288, ConstantExpr::create(static_cast<int64_t>(8) * 1024 * 1024 * 1024, Expr::Int64)); // totalGlobalMem: 8GB
    wos->write(296, ConstantExpr::create(48 * 1024, Expr::Int64)); // sharedMemPerBlock: 48KB
    wos->write(304, ConstantExpr::create(65536, Expr::Int32)); // 65536 registers per block
    wos->write(308, ConstantExpr::create(32, Expr::Int32)); // warpSize: 32
    wos->write(312, ConstantExpr::create(1048576, Expr::Int64)); // memPitch: 1MB
    wos->write(320, ConstantExpr::create(1024, Expr::Int32)); // maxThreadsPerBlock
    wos->write(324, ConstantExpr::create(1024, Expr::Int32)); // maxThreadsDim X
    wos->write(328, ConstantExpr::create(1024, Expr::Int32)); // maxThreadsDim Y
    wos->write(332, ConstantExpr::create(64, Expr::Int32));   // maxThreadsDim Z
    wos->write(336, ConstantExpr::create(2147483647, Expr::Int32)); // maxGridSize X
    wos->write(340, ConstantExpr::create(65535, Expr::Int32));      // maxGridSize Y
    wos->write(344, ConstantExpr::create(65535, Expr::Int32));      // maxGridSize Z
    wos->write(348, ConstantExpr::create(1500000, Expr::Int32)); // clockRate: 1.5 GHz
    wos->write(352, ConstantExpr::create(64 * 1024, Expr::Int64)); // totalConstMem: 64 KB
    wos->write(360, ConstantExpr::create(9, Expr::Int32)); // Compute Capability major 9.x
    wos->write(364, ConstantExpr::create(0, Expr::Int32)); // Compute Capability minor 9.0
    wos->write(388, ConstantExpr::create(90, Expr::Int32));       // multiProcessorCount
    wos->write(624, ConstantExpr::create(2048, Expr::Int32));     // maxThreadsPerMultiProcessor
    wos->write(640, ConstantExpr::create(65536, Expr::Int64));    // sharedMemPerMultiprocessor
    wos->write(648, ConstantExpr::create(65536, Expr::Int32));    // regsPerMultiprocessor
    wos->write(712, ConstantExpr::create(32, Expr::Int32));       // maxBlocksPerMultiProcessor

    for (int i = 0; i < 16; i++) {
      wos->write(256 + i, ConstantExpr::create(0, Expr::Int8));
    }

    for (int i = 0; i < 8; i++) {
      wos->write(272 + i, ConstantExpr::create(0, Expr::Int8));
    }

    wos->write(280, ConstantExpr::create(0, Expr::Int32));

    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("at4cuda26getCurrentDeviceProperties") != std::string::npos || callable->getName().find("at4cuda19getDeviceProperties") != std::string::npos || callable->getName().find("cudaGetDeviceProperties") != std::string::npos) {
    MemoryObject *mo = memory->allocate(848, false, false, &state, state.prevPC->inst, 8);
    ObjectState *os = bindObjectInState(state, mo, false);
    // Initialize properties with common values
    os->write(288, ConstantExpr::create(static_cast<int64_t>(8) * 1024 * 1024 * 1024, Expr::Int64)); // totalGlobalMem: 8GB
    os->write(296, ConstantExpr::create(48 * 1024, Expr::Int64)); // sharedMemPerBlock: 48KB
    os->write(304, ConstantExpr::create(65536, Expr::Int32)); // 65536 registers per block
    os->write(308, ConstantExpr::create(32, Expr::Int32)); // warpSize: 32
    os->write(312, ConstantExpr::create(1048576, Expr::Int64)); // memPitch: 1MB
    os->write(320, ConstantExpr::create(1024, Expr::Int32)); // maxThreadsPerBlock
    os->write(324, ConstantExpr::create(1024, Expr::Int32)); // maxThreadsDim X
    os->write(328, ConstantExpr::create(1024, Expr::Int32)); // maxThreadsDim Y
    os->write(332, ConstantExpr::create(64, Expr::Int32));   // maxThreadsDim Z
    os->write(336, ConstantExpr::create(2147483647, Expr::Int32)); // maxGridSize X
    os->write(340, ConstantExpr::create(65535, Expr::Int32));      // maxGridSize Y
    os->write(344, ConstantExpr::create(65535, Expr::Int32));      // maxGridSize Z
    os->write(348, ConstantExpr::create(1500000, Expr::Int32)); // clockRate: 1.5 GHz
    os->write(352, ConstantExpr::create(64 * 1024, Expr::Int64)); // totalConstMem: 64 KB
    os->write(360, ConstantExpr::create(9, Expr::Int32)); // Compute Capability major 9.x
    os->write(364, ConstantExpr::create(0, Expr::Int32)); // Compute Capability minor   9.0
    os->write(388, ConstantExpr::create(90, Expr::Int32));       // multiProcessorCount
    os->write(624, ConstantExpr::create(2048, Expr::Int32));     // maxThreadsPerMultiProcessor
    os->write(640, ConstantExpr::create(65536, Expr::Int64));    // sharedMemPerMultiprocessor
    os->write(648, ConstantExpr::create(65536, Expr::Int32));    // regsPerMultiprocessor
    os->write(712, ConstantExpr::create(32, Expr::Int32));       // maxBlocksPerMultiProcessor

    for (int i = 0; i < 16; i++) {
        os->write(256 + i, ConstantExpr::create(0, Expr::Int8));
    }

    for (int i = 0; i < 8; i++) {
        os->write(272 + i, ConstantExpr::create(0, Expr::Int8));
    }

    // Initialize LUID device node mask
    os->write(280, ConstantExpr::create(0, Expr::Int32));

    bindLocal(target, state, mo->getBaseExpr());
    return;
  }

  if (callable->getName().find("at26assert_no_internal_overlapERKNS_10TensorBase") != std::string::npos || callable->getName().find("at17assert_no_overlapERKNS_10TensorBase") != std::string::npos) {
    return;
  }

  if (callable->getName().find("19maybe_wrap_dim_slow") != std::string::npos) {
    terminateStateOnProgramError(state, "input tensor dim invalid", StateTerminationType::InvalidInput);
    return;
  }

  if (callable->getName().find("at6native20canUse32BitIndexMathERKNS_10TensorBase") != std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at6native20canUse32BitIndexMathERKNS_10TensorBase: do not find the memory object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op.first;
    std::string symName = getSymName(state, mo, arguments[0]);
    if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
      ref<Expr> size = state.symbolicArrayMap[symName]->size;
      ref<Expr> condition = UleExpr::create(size, ZExtExpr::create(arguments[1], size->getWidth()));
      StatePair branches = fork(state, condition, true, BranchType::Conditional);
      ExecutionState *bound = branches.first;
      if (bound) {
        bindLocal(target, *bound, ConstantExpr::create(1, Expr::Bool));
      }

      ExecutionState *unbound = branches.second;
      if(unbound) {
        bindLocal(target, *unbound, ConstantExpr::create(0, Expr::Bool));
      }
    } else {
      terminateStateOnProgramError(state, "at6native20canUse32BitIndexMathERKNS_10TensorBase: do not find the symbolic array", StateTerminationType::ReportError);
      return;
    }
    return;
  }

  if (callable->getName().find("at19rspmm_forward_checkEPKcRKNS_9TensorArgES4_S4_S4_S4_") != std::string::npos || callable->getName().find("at15checkAllSameGPUEPKcN3c108ArrayRefINS_9TensorArgEEE") != std::string::npos) {
    return;
  }

  if (callable->getName().find("at7ind2ptrERKNS_6TensorEi") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at7ind2ptrERKNS_6TensorEi: do not find the tensor object", StateTerminationType::ReportError);
      return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at7ind2ptrERKNS_6TensorEi: do not find the tensor object", StateTerminationType::ReportError);
      return;
    }

    ref<Expr> size = AddExpr::create(arguments[2], ConstantExpr::create(1, arguments[2]->getWidth()));
    const MemoryObject *sourceMO = op1.first;
    const MemoryObject *destMO = op0.first;

    SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "at7ind2ptrERKNS_6TensorEi", "ind2ptr_tensor_");
    destSymmo->size = size;
    destSymmo->dimensionSize[0] = size;
    destSymmo->shapeSize = ConstantExpr::create(1, Expr::Int64);
    destSymmo->scalarType = ConstantExpr::create(4, Expr::Int8);
    destSymmo->elementSize = ConstantExpr::create(8, Expr::Int64);

    std::string sourceName = getSymName(state, sourceMO, arguments[1]);
    if (state.symbolicArrayMap.find(sourceName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceName];
      destSymmo->maxVal = sourceSymmo->size;
    }
    destSymmo->minVal = ConstantExpr::create(0, Expr::Int32);
    return;
  }

  if (callable->getName().find("cudaSetDevice") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Bool));
    return;
  }

  if (callable->getName().find("omp_get_wtime") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Int64));
    return;
  }

  if (callable->getName().find("at13globalContextEv") != std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    int argSize = kmodule->targetData->getTypeAllocSize(returnType->getPointerElementType());
    MemoryObject *contextMo =
          memory->allocate(argSize, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);
    ObjectState *os = bindObjectInState(state, contextMo, false);
    os->initializeToZero();
    bindLocal(target, state, contextMo->getBaseExpr());  
    return;
  } 

  if (callable->getName().find("at4cuda6detail23getDefaultCUDAGenerator") != std::string::npos) {
    MemoryObject *generatorMo =
          memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);
    ObjectState *generatorOS = bindObjectInState(state, generatorMo, false);
    
    MemoryObject *implMo =
          memory->allocate(80, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);
    ObjectState *implOS = bindObjectInState(state, implMo, false);
    implOS->initializeToZero();
    implOS->write(8, ConstantExpr::create(1, Expr::Int32));
    generatorOS->write(0, implMo->getBaseExpr());
    bindLocal(target, state, generatorMo->getBaseExpr());
    return;
  }

  if (callable->getName().find("at17CUDAGeneratorImpl11device_type") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Int8));
    return;
  }

  if (callable->getName().find("3c1013GeneratorImpl6device") != std::string::npos) {
    bindLocal(target, state, ConcatExpr::create(ConstantExpr::create(1, Expr::Int8), ConstantExpr::create(1, Expr::Int8)));
    return;
  }

  if (callable->getName().find("at17CUDAGeneratorImpl17philox_cuda_state") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at17CUDAGeneratorImpl17philox_cuda_state: do not find the hiloxCudaState object", StateTerminationType::ReportError);
        return;
    }
    ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
    wos->write(0, ConstantExpr::create(0, Expr::Int64));
    wos->write(8, arguments[2]);
    return;
  }

  if (callable->getName().find("SymIntC1ENS_13intrusive_ptrINS_11SymNodeImplENS_6detail34intrusive_target_default_null_type") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "c10::SymInt::SymInt(c10::intrusive_ptr<c10::SymNodeImpl> ptr): do not find the SymInt object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at7ind2ptrERKNS_6TensorEi: do not find the intrusive_ptr object", StateTerminationType::ReportError);
        return;
    }

    ref<Expr> symNodeImplPtr = op1.second->read(0, 64);
    ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
    wos->write(0, symNodeImplPtr);
    return;
  }

  if (callable->getName().find("2at4cuda9warp_size") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(32, Expr::Int32));
    return;
  }

  if (callable->getName().find("c1012WarningUtils14get_warnAlways") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Bool));
    return;
  }

  if (callable->getName().find("at15checkScalarTypeEPKcRKNS_9TensorArgEN3c1010ScalarType") != std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at::checkScalarType: do not find the TensorArg object", StateTerminationType::ReportError);
        return;
    }
    ref<Expr> address = op.second->read(0, 64);

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), address, op1, success) || !success) {
      terminateStateOnProgramError(state, "at::checkScalarType: do not find the tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op1.first;
    std::string symName = getSymName(state, mo, arguments[0]);
    if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
      addConstraint(state, EqExpr::create(ZExtExpr::create(symmo->scalarType, arguments[2]->getWidth()), arguments[2]));
      return;
    } else {
      terminateStateOnProgramError(state, "at::checkScalarType: do not find symbolic tensor object", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("TensorIteratorBase5numel")!=std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    int width = returnType->getIntegerBitWidth();
    if (!state.inputIterator->numel) {
      computeTensorIteratorShape(state);
      ref<Expr> numel = ConstantExpr::create(1, width);
      for (auto &e: state.inputIterator->shape) {
        numel = MulExpr::create(numel, ZExtExpr::create(e, width));
      }
      state.inputIterator->numel = numel;
    }
    bindLocal(target, state, ZExtExpr::create(state.inputIterator->numel, width));
    return;
  }

  if (callable->getName().find("TensorIteratorBase22can_use_32bit_indexing")!=std::string::npos) {
    for (auto name: state.inputIterator->tensors) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[name];
      ref<Expr> maxOffset = ZExtExpr::create(symmo->size, Expr::Int64);
      if (!symmo->elementSize.isNull()) {
        maxOffset = MulExpr::create(maxOffset, ZExtExpr::create(symmo->elementSize, Expr::Int64));
      }
      addConstraint(state, UleExpr::create(maxOffset, ConstantExpr::create((1ULL << 31) - 1, Expr::Int64)));
    }
    bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
    return;
  }

  if (callable->getName().find("TensorIteratorBase8data_ptr")!=std::string::npos) {
    if (auto CE = dyn_cast<ConstantExpr>(arguments[1])) {
      int index = CE->getZExtValue();
      std::string name = state.inputIterator->tensors[index];
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[name];

      if (auto sizeCE = dyn_cast<ConstantExpr>(symmo->size)) {
        int size = sizeCE->getZExtValue();
        if (size == 0) {
          bindLocal(target, state, ConstantExpr::create(0, Expr::Int64));
          return;
        }
      }
      bindLocal(target, state, ConstantExpr::createPointer(symmo->arrayAddress));
      return;
    } else {
      terminateStateOnProgramError(state, "TensorIteratorBase8data_ptr: index is not constant", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("TensorIteratorBase13is_contiguous")!=std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
    return;
  }

  if (callable->getName().find("TensorIterator8unary_opERNS_10TensorBase")!=std::string::npos) {
    if (!isa<ConstantExpr>(arguments[0])) {
      terminateStateOnProgramError(state, "TensorIterator address is not constant", StateTerminationType::ReportError);
        return;
    }
    auto aCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t address = aCE->getZExtValue();
    TensorIterator *iterator = new TensorIterator(address, nullptr, nullptr, ConstantExpr::create(2, Expr::Int32), 1);
    
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op0, success) || !success) {
      terminateStateOnProgramError(state, "TensorIterator8unary_opERNS_10TensorBase: do not find output tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op1, success) || !success) {
      terminateStateOnProgramError(state, "TensorIterator8unary_opERNS_10TensorBase: do not find input tensor objec", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo0 = op0.first;
    iterator->addTensor(mo0->name, false);
    const MemoryObject *mo1 = op1.first;
    iterator->addTensor(mo1->name, true);
    state.inputIterator = iterator;
    return;
  }

  if (callable->getName().find("TensorIterator20borrowing_nullary_opERKNS_10TensorBase") != std::string::npos) {
    if (!isa<ConstantExpr>(arguments[0])) {
      terminateStateOnProgramError(state, "TensorIterator address is not constant", StateTerminationType::ReportError);
        return;
    }
    auto aCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t address = aCE->getZExtValue();
    TensorIterator *iterator = new TensorIterator(address, nullptr, nullptr, ConstantExpr::create(1, Expr::Int32), 1);
    
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "TensorIterator20borrowing_nullary_opERKNS_10TensorBase: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op.first;
    iterator->addTensor(mo->name, false);
    state.inputIterator = iterator;
    return;
  }

  if (callable->getName().find("cudaGetDevice") != std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "cudaGetDevice: do not find target object", StateTerminationType::ReportError);
        return;
    }

    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    wos->write(0, ConstantExpr::create(0, Expr::Int32));
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("cudaStreamSynchronize") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("at6native12quantize_valIN3c") != std::string::npos) {
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();
    ref<Expr> invScaleExpr;
    if (isa<ConstantExpr>(arguments[0])) {
      float scaleVal = ConstantExpr::getFloatFromConstantExpr(ZExtExpr::create(arguments[0], Expr::Int32));
      float inv_scale = 1.0f / scaleVal;
      uint32_t inv_scale_bits;
      std::memcpy(&inv_scale_bits, &inv_scale, sizeof(inv_scale));
      invScaleExpr = ConstantExpr::alloc(inv_scale_bits, Expr::Int32, true, true);
    } else {
      invScaleExpr = SDivExpr::create(ConstantExpr::create(1, Expr::Int32), ZExtExpr::create(arguments[0], Expr::Int32));
    }

    // value * inv_scale
    ref<Expr> scaledValue = MulExpr::create(arguments[2], invScaleExpr);

    ref<Expr> half = ConstantExpr::alloc(0x3f000000, Expr::Int32, false, true); // 0.5f
    ref<Expr> rounded = AddExpr::create(scaledValue, half);
    // Convert to int64: truncate
    ref<Expr> roundedInt = ZExtExpr::create(rounded, Expr::Int64);

    ref<Expr> qvalue = AddExpr::create(arguments[1], roundedInt);

    // Clamp to qmin, qmax
    ref<Expr> qminExpr = ConstantExpr::alloc(0, Expr::Int64);
    ref<Expr> qmaxExpr = ConstantExpr::alloc(255, Expr::Int64);
    ref<Expr> clampedLow = SelectExpr::create(
        UltExpr::create(qvalue, qminExpr), qminExpr, qvalue);
    ref<Expr> clamped = SelectExpr::create(
        UgtExpr::create(clampedLow, qmaxExpr), qmaxExpr, clampedLow);
    bindLocal(target, state, ZExtExpr::create(clamped, returnType->getIntegerBitWidth()));
    return;
  }

  if (callable->getName().find("20has_internal_overlapERKNS_10TensorBase") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("25assert_no_partial_overlapERKNS_10TensorBase") != std::string::npos) {
    return;
  }

  if (callable->getName().find("at7Context36deterministicFillUninitializedMemoryEv") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
    return;
  }

  if (callable->getName().find("at7Context23deterministicAlgorithms") != std::string::npos) {
    std::string symName = "context.deterministicAlgorithm";
    MemoryObject *mo =
          memory->allocate(1, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.pc->inst,
                          /*alignment=*/8);    
    mo->setName(symName);
    const Array *array = arrayCache.CreateArray(symName, mo->size, true);
    bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);
    ref<Expr> sizeExpr = Expr::createTempRead(array, mo->size*8);

    addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(0, sizeExpr->getWidth())));
    addConstraint(state, UleExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
    bindLocal(target, state, ZExtExpr::create(sizeExpr, Expr::Bool));
    return;
  }

  if (callable->getName().find("at4cuda3cub18mask_exclusive_sum") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at4cuda3cub18mask_exclusive_sum: do not find first tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at4cuda3cub18mask_exclusive_sum: do not find second tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo0 = op0.first;
    std::string symName0 = getSymName(state, mo0, arguments[0]);
    const MemoryObject *mo1 = op1.first;
    std::string symName1 = getSymName(state, mo1, arguments[1]);

    if (state.symbolicArrayMap.find(symName0) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo0 = state.symbolicArrayMap[symName0];
      if (symmo0->size!=arguments[2]) {
        addConstraint(state, EqExpr::create(arguments[2], ZExtExpr::create(symmo0->size, arguments[2]->getWidth())));
      }
    }

    if (state.symbolicArrayMap.find(symName1) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo1 = state.symbolicArrayMap[symName1];
      if (symmo1->size!=arguments[2]) {
        addConstraint(state, EqExpr::create(arguments[2], ZExtExpr::create(symmo1->size, arguments[2]->getWidth())));
      }
    }
    return;
  }

  if (callable->getName().find("cudaFuncGetAttributes") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
    return;
  }

  if (callable->getName().find("c104cuda29c10_cuda_check_implementation") != std::string::npos) {
    return;
  }

  if (callable->getName().find("TensorIteratorBase13is_cpu_scalar") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("at14TensorIteratorD2Ev") != std::string::npos) {
    return;
  }

  if (callable->getName().find("at14namedinference21broadcast_to_outnamesERKNS_6Tensor") != std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at14namedinference21broadcast_to_outnamesERKNS_6Tensor: do not find vector object", StateTerminationType::ReportError);
        return;
    }
    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    wos->write(0, ConstantExpr::create(0, Expr::Int64));
    wos->write(8, ConstantExpr::create(0, Expr::Int64));
    wos->write(16, ConstantExpr::create(0, Expr::Int64));
    return;
  }

  if (callable->getName().find("WarningC1ESt7variantIJNS0_11UserWarningENS0_18DeprecationWarning") != std::string::npos) {
    return;
  }

  if (callable->getName().find("at4_ops3max4callERKNS_6Tensor") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops3max4callERKNS_6Tensor: do not find the return tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops3max4callERKNS_6Tensor: do not find the source tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
    const MemoryObject *destMO = op0.first;
    if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "at4_ops3max4callERKNS_6Tensor", "max_tensor_");
      destSymmo->size = ConstantExpr::create(1, Expr::Int64);
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->shapeSize = ConstantExpr::create(1, Expr::Int64);
      std::map<int, ref<Expr>> dimensionSize;
      dimensionSize[0] = ConstantExpr::create(1, Expr::Int64);
      destSymmo->dimensionSize = dimensionSize;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;
    } else {
      terminateStateOnProgramError(state, "at4_ops3max4callERKNS_6Tensor: do not find the source symbolic object", StateTerminationType::ReportError);
      return;
    }
    return;
  }

  if (callable->getName().find("at4_ops3min4callERKNS_6TensorE") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops3min4callERKNS_6Tensor: do not find the return tensor object", StateTerminationType::ReportError);
      return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops3min4callERKNS_6Tensor: do not find the source tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
    const MemoryObject *destMO = op0.first;
    if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "at4_ops3min4callERKNS_6Tensor", "max_tensor_");
      
      destSymmo->size = ConstantExpr::create(1, Expr::Int64);
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->shapeSize = ConstantExpr::create(1, Expr::Int64);
      std::map<int, ref<Expr>> dimensionSize;
      dimensionSize[0] = ConstantExpr::create(1, Expr::Int64);
      destSymmo->dimensionSize = dimensionSize;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;
    } else {
      terminateStateOnProgramError(state, "at4_ops3min4callERKNS_6Tensor: do not find the source symbolic object", StateTerminationType::ReportError);
      return;
    }
    return;
  }

  if (callable->getName().find("2at6Tensor4item") != std::string::npos && callable->getName().find("EET_v") != std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "at6Tensor4item: do not find the tensor object", StateTerminationType::ReportError);
        return;
    }
    const MemoryObject *mo = op.first;
    std::string symName = getSymName(state, mo, arguments[0]);
    if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
      llvm::Type *returnType = callable->getFunctionType()->getReturnType();
      int returnBits = returnType->getPrimitiveSizeInBits();

      std::string itemName = symName + ".item";
      MemoryObject *itemMo =
        memory->allocate(returnBits/8, /*isLocal=*/false, /*isGlobal=*/false,
                        &state, /*allocSite=*/state.prevPC->inst,
                        /*alignment=*/8);          
      ref<Expr> itemExpr = symbolizeIndex(state, itemName, itemMo);
      bindLocal(target, state, itemExpr);
      return;
    } else {
      terminateStateOnProgramError(state, "at6Tensor4item: do not find the tensor symbolic object", StateTerminationType::ReportError);
      return;
    }
  }

  if (callable->getName().find("at6native23direct_copy_kernel_cudaERNS_18TensorIteratorBase") != std::string::npos) {
    std::string outputName = state.inputIterator->tensors[0];
    std::string inputName = state.inputIterator->tensors[1];

    if (state.symbolicArrayMap.find(inputName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[inputName];
      SymArrayMemoryObject *destSymmo;
      if (state.symbolicArrayMap.find(outputName) != state.symbolicArrayMap.end()) {
        destSymmo = state.symbolicArrayMap[outputName];
      } else {
        terminateStateOnProgramError(state, "at6native23direct_copy_kernel_cudaERNS_18TensorIteratorBase: do not find the symbolic output tensor", StateTerminationType::ReportError);
        return;
      }
      destSymmo->size = sourceSymmo->size;
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->shapeSize = sourceSymmo->shapeSize;
      destSymmo->dimensionSize = sourceSymmo->dimensionSize;
      destSymmo->strides = sourceSymmo->strides;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;
      return;
    } else {
      terminateStateOnProgramError(state, "at6native23direct_copy_kernel_cudaERNS_18TensorIteratorBase: do not find the symbolic input tensor", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("at15get_tensor_baseERKNS_6Tensor") != std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("3c104warnERKNS_7WarningE") != std::string::npos) {
    return;
  }

  if (callable->getName().find("at18TensorIteratorBase19num_output_elements") != std::string::npos) {
    std::string outputName = state.inputIterator->tensors[0];
    llvm::Type *returnType = callable->getFunctionType()->getReturnType();

    if (state.symbolicArrayMap.find(outputName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[outputName];
      bindLocal(target, state, ZExtExpr::create(sourceSymmo->size, returnType->getIntegerBitWidth()));
      return;
    } else {
      terminateStateOnProgramError(state, "at18TensorIteratorBase19num_output_elements: do not find the symbolic output tensor", StateTerminationType::ReportError);
        return;
    }
    return;
  }

  if (callable->getName().find("at4_ops13scalar_tensor4callERKN3c106ScalarESt8optionalINS2_10ScalarTypeEES6_INS2_6LayoutEES6_INS2_6Device") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at::_ops::scalar_tensor::call: do not find return tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *destMO = op0.first;
    SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "at::_ops::scalar_tensor::call", "scalar_tensor_");

    ref<Expr> scalarType = ExtractExpr::create(arguments[2], 0, 8);
    destSymmo->scalarType = scalarType;
    std::map<int, ref<Expr>> dimensionSize;
    dimensionSize[0] = ConstantExpr::create(1, Expr::Int64);
    destSymmo->dimensionSize = dimensionSize;
    destSymmo->size = ConstantExpr::create(1, Expr::Int64);
    destSymmo->shapeSize = ConstantExpr::create(0, Expr::Int64);
    return;
  }

  if (callable->getName().find("at4_ops6arange4callERKN3c106ScalarESt8optionalINS2_10ScalarTypeEES6_INS2_6LayoutEES6_INS2_6DeviceEES6_IbE") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at::_ops::arange::call: do not find return tensor object", StateTerminationType::ReportError);
      return;
    }

    const MemoryObject *destMO = op0.first;
    SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "at::_ops::arange::call", "arange_tensor_");

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at::_ops::arange::call: do not find scalar object", StateTerminationType::ReportError);
      return;
    }
    const ObjectState *scalarOS = op1.second;
    ref<Expr> value = scalarOS->read(16, Expr::Int64);

    ref<Expr> scalarType = ExtractExpr::create(arguments[2], 0, 8);
    destSymmo->scalarType = scalarType;
    std::map<int, ref<Expr>> dimensionSize;
    dimensionSize[0] = value;
    destSymmo->dimensionSize = dimensionSize;
    destSymmo->size = value;
    destSymmo->shapeSize = ConstantExpr::create(1, Expr::Int64);
    return;
  }

  if (callable->getName().find("at4_ops3all4callERKNS_6TensorE") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops3all4callERKNS_6TensorE: do not find return tensor object", StateTerminationType::ReportError);
        return;
    }


    const MemoryObject *destMO = op0.first;


    SymArrayMemoryObject *destSymmo;
    ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t destAddress = destAddressCE->getZExtValue();
    if (destAddress != destMO->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        std::string destMName = state.symNames[destAddress];
        if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "at4_ops3all4callERKNS_6TensorE: do not find target tensor object", StateTerminationType::ReportError);
        return;
        }
        destSymmo = state.symbolicArrayMap[destMName];
      } else {
        unsigned id = 0;
        std::string uniqueName = "ops_all_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "ops_all_tensor_" + llvm::utostr(++id);
        }
        const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
        destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = destSymmo;
        state.symNames[destAddress] = uniqueName;
        state.symAddressMap[destMO->address] = destAddress;
      }
    } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
      unsigned id = 0;
      std::string uniqueName = "ops_all_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "ops_all_tensor_" + llvm::utostr(++id);
      }
      destMO->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
      bindObjectInState(state, destMO, false, array);
      state.addSymbolic(destMO, array);
      destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
      state.symbolicArrayMap[destMO->name] = destSymmo;
    } else {
      destSymmo = state.symbolicArrayMap[destMO->name];
    }

    destSymmo->scalarType = ConstantExpr::create(11, Expr::Int64);;
    std::map<int, ref<Expr>> dimensionSize;
    dimensionSize[0] = ConstantExpr::create(1, Expr::Int64);
    destSymmo->dimensionSize = dimensionSize;
    destSymmo->size = ConstantExpr::create(1, Expr::Int64);
    destSymmo->shapeSize = ConstantExpr::create(0, Expr::Int64);
    return;
  }

  if (callable->getName().find("at6native19resize_output_checkERKNS_6TensorEN3c108ArrayRef") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "resize_output_check: do not find target tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *mo = op0.first;
    std::string symName = getSymName(state, mo, arguments[0]);
    if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
      if (auto shapeSizeExpr = dyn_cast<ConstantExpr>(arguments[2])) {
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        ObjectPair op1;
        if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
          terminateStateOnProgramError(state, "resize_output_check: do not find the shape array object", StateTerminationType::ReportError);
        return;
        }

        const ObjectState *shapeArrayOS = op1.second;
        ref<Expr> cond = EqExpr::create(arguments[2], ZExtExpr::create(symmo->shapeSize, arguments[2]->getWidth()));
        for (uint64_t i = 0; i < shapeSize; ++i) {
          ref<Expr> element = shapeArrayOS->read(i*8, 64);
          if (symmo->dimensionSize.find(i) == symmo->dimensionSize.end()) {
            terminateStateOnProgramError(state, "resize_output_check: tensor dimsize does not exist", StateTerminationType::ReportError);
        return;
          }
          cond = AndExpr::create(cond, EqExpr::create(element, ZExtExpr::create(symmo->dimensionSize[i], Expr::Int64)));
        }
        cond = optimizer.optimizeExpr(cond, false);
        if (auto condCE = dyn_cast<ConstantExpr>(cond)) {
          bindLocal(target, state, condCE);
          return;
        }
        addConstraint(state, cond);
        bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
        return;
      } else {
        terminateStateOnProgramError(state, "resize_output_check: array size is not a constant", StateTerminationType::ReportError);
        return;
      }
    } else {
      terminateStateOnProgramError(state, "resize_output_check: do not find tensor symbolic object", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("at16toAccumulateTypeEN3c1010ScalarType") != std::string::npos) {
    ref<Expr> acutype = getAccumalateTensorType(arguments[0]);
    bindLocal(target, state, acutype);
    return;
  }

  if (callable->getName().find("at6native26_select_batch_norm_backendERKNS_6Tensor") != std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("at4_ops9ge_Scalar4callERKNS_6TensorERKN3c106Scalar") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops9ge_Scalar4callERKNS_6TensorERKN3c106Scalar: do not find return tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops9ge_Scalar4callERKNS_6TensorERKN3c106Scalar: do not find source tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
    const MemoryObject *destMO = op0.first;

    if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
      terminateStateOnProgramError(state, "at4_ops9ge_Scalar4callERKNS_6TensorERKN3c106Scalar: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
    }

    SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
    SymArrayMemoryObject *destSymmo;
    ref<ConstantExpr> destAddressCE = dyn_cast<ConstantExpr>(arguments[0]);
    uint64_t destAddress = destAddressCE->getZExtValue();
    if (destAddress != destMO->address) {
      if (state.symNames.find(destAddress)!=state.symNames.end()) {
        std::string destMName = state.symNames[destAddress];
        if (state.symbolicArrayMap.find(destMName) == state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "at4_ops9ge_Scalar4callERKNS_6TensorERKN3c106Scalar: do not find target tensor object", StateTerminationType::ReportError);
        return;
        }
        destSymmo = state.symbolicArrayMap[destMName];
      } else {
        unsigned id = 0;
        std::string uniqueName = "ge_Scalar_tensor_0";
        while (!state.arrayNames.insert(uniqueName).second) {
          uniqueName = "ge_Scalar_tensor_" + llvm::utostr(++id);
        }
        const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
        destSymmo = new SymArrayMemoryObject(destAddress, array, nullptr, nullptr);
        state.symbolicArrayMap[uniqueName] = destSymmo;
        state.symNames[destAddress] = uniqueName;
        state.symAddressMap[destMO->address] = destAddress;
      }
    } else if (state.symbolicArrayMap.find(destMO->name) == state.symbolicArrayMap.end()) {
      unsigned id = 0;
      std::string uniqueName = "ge_Scalar_tensor_0";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "ge_Scalar_tensor_" + llvm::utostr(++id);
      }
      destMO->setName(uniqueName);
      const Array *array = arrayCache.CreateArray(uniqueName, destMO->size);
      bindObjectInState(state, destMO, false, array);
      state.addSymbolic(destMO, array);
      destSymmo = new SymArrayMemoryObject(destMO->address, array, nullptr, nullptr);
      state.symbolicArrayMap[destMO->name] = destSymmo;
    } else {
      destSymmo = state.symbolicArrayMap[destMO->name];
    }

    destSymmo->dimensionSize = sourceSymmo->dimensionSize;
    destSymmo->strides = sourceSymmo->strides;
    destSymmo->shapeSize = sourceSymmo->shapeSize;
    destSymmo->sizeName = sourceSymmo->sizeName;
    destSymmo->size = sourceSymmo->size;
    destSymmo->scalarType = ConstantExpr::create(11, Expr::Int8);
    destSymmo->maxVal = sourceSymmo->maxVal;
    destSymmo->minVal = sourceSymmo->minVal;
    destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
    destSymmo->isFloat = sourceSymmo->isFloat;
    return;
  }

  if (callable->getName().find("at10TensorBase8toStringB5cxx11") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at10TensorBase8toStringB5cxx11: do not find string object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at10TensorBase8toStringB5cxx11: do not find tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);

    std::string s = sourceMO->name;
    unsigned len = s.length();
    for (unsigned i = 0; i < len; ++i) {
      wos->write8(i, s[i]);
    }
    wos->write8(len, '\0'); 
    return;
  }

  if (callable->getName().find("3c1015SmallVectorBaseIjE8grow_podEPKvmm") != std::string::npos) { // TODO
    return;
  }

  if (callable->getName().find("at18TensorIteratorBase13is_trivial_1d") != std::string::npos) {
    computeTensorIteratorShape(state);
    if (state.inputIterator->shape.empty()) {
      bindLocal(target, state, ConstantExpr::create(0, Expr::Bool));
    } else {
      bindLocal(target, state, EqExpr::create(ConstantExpr::create(state.inputIterator->shape.size(), Expr::Int64), ConstantExpr::create(1, Expr::Int64)));
    }
    return;
  }

  if (callable->getName().find("10TensorBase2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormat") != std::string::npos) {
    bool success;
    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "TensorBase::to: do not find source tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op2;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
      terminateStateOnProgramError(state, "TensorBase::to: do not find target tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
    const MemoryObject *destMO = op2.first;
    std::string destSymName = getSymName(state, destMO, arguments[0]);
    if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "TensorBase::to", "tensorBase_to_");     
      destSymmo->size = sourceSymmo->size;
      destSymmo->sizeName = sourceSymmo->sizeName;
      destSymmo->dimensionSize = sourceSymmo->dimensionSize;
      destSymmo->strides = sourceSymmo->strides;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->shapeSize = sourceSymmo->shapeSize;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;

      ref<Expr> scalarType = ExtractExpr::create(arguments[2], 16, 16);
      destSymmo->scalarType = scalarType;
      unsigned size = getTensorTypeBytes(scalarType);
      if (size > 0)
        destSymmo->elementSize = ConstantExpr::create(size, Expr::Int64);
      else {
        ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(scalarType);
        if (readExpr) {
          std::string readName = readExpr->updates.root->name;
          std::string suffix = ".scalar_type";
          if (readName.size() >= suffix.size()) {
            if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
              std::string tensorName = readName.substr(0, readName.size() - suffix.size());
              if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                if (originalSymmo->elementSize) {
                  destSymmo->elementSize = originalSymmo->elementSize;
                }
              }
            }
          }
        }
      }
      return;
    } else {
      terminateStateOnProgramError(state, "TensorBase::to: do not find target tensor object", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("2at4_ops8to_dtype4callERKNS_6TensorEN3c1010ScalarTypeEbbSt8optionalINS5_12MemoryFormatEE") != std::string::npos) {
    bool success;
    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at::_ops::to_dtype::call: do not find source tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op2;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op2, success) || !success) {
      terminateStateOnProgramError(state, "at::_ops::to_dtype::call: do not find target tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
    const MemoryObject *destMO = op2.first;
    std::string destSymName = getSymName(state, destMO, arguments[0]);
    if (state.symbolicArrayMap.find(sourceSymName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
      SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "TensorBase::to", "tensorBase_to_");     
      destSymmo->size = sourceSymmo->size;
      destSymmo->sizeName = sourceSymmo->sizeName;
      destSymmo->dimensionSize = sourceSymmo->dimensionSize;
      destSymmo->strides = sourceSymmo->strides;
      destSymmo->shapeSize = sourceSymmo->shapeSize;
      destSymmo->scalarType = arguments[2];
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;

      unsigned size = getTensorTypeBytes(arguments[2]);
      if (size > 0)
        destSymmo->elementSize = ConstantExpr::create(size, Expr::Int64);
      else {
        ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(arguments[2]);
        if (readExpr) {
          std::string readName = readExpr->updates.root->name;
          std::string suffix = ".scalar_type";
          if (readName.size() >= suffix.size()) {
            if (readName.compare(readName.size() - suffix.size(), suffix.size(), suffix) == 0) {
              std::string tensorName = readName.substr(0, readName.size() - suffix.size());
              if (state.symbolicArrayMap.find(tensorName) != state.symbolicArrayMap.end()) {
                SymArrayMemoryObject *originalSymmo = state.symbolicArrayMap[tensorName];
                if (originalSymmo->elementSize) {
                  destSymmo->elementSize = originalSymmo->elementSize;
                }
              }
            }
          }
        }
      }
      return;
    } else {
      terminateStateOnProgramError(state, "TensorBase::to: do not find target tensor object", StateTerminationType::ReportError);
        return;
    }
  }

  if (callable->getName().find("at4_ops11squeeze_dim4callERKNS_6TensorEl") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops11squeeze_dim4callERKNS_6TensorEl: do not find source tensor object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "at4_ops11squeeze_dim4callERKNS_6TensorEl: do not find target tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op1.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[1]);
    const MemoryObject *destMO = op0.first;

    if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
      terminateStateOnProgramError(state, "at4_ops11squeeze_dim4callERKNS_6TensorEl: do not find source tensor symbolic object", StateTerminationType::ReportError);
        return;
    }

    SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
    SymArrayMemoryObject *destSymmo = createSymArray(state, arguments[0], destMO, "at4_ops11squeeze_dim4callERKNS_6TensorEl", "torch_squeeze_tensor_");

    if (auto dimIndexExpr = dyn_cast<ConstantExpr>(arguments[2])) {
      if (!sourceSymmo->shapeSize.isNull() && isa<ConstantExpr>(sourceSymmo->shapeSize)) {
        auto shapeSizeExpr = dyn_cast<ConstantExpr>(sourceSymmo->shapeSize);
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        uint64_t dimIndex = dimIndexExpr->getZExtValue();

        bool remove = false;
        for (uint64_t i = 0; i < shapeSize; ++i) {
          if (i == dimIndex) {
            if (auto dimSizeExpr = dyn_cast<ConstantExpr>(sourceSymmo->dimensionSize[i])) {
              uint64_t dimSize = dimSizeExpr->getZExtValue();
              if (dimSize == 1) {
                remove = true;
                continue;
              }
            }
          } else if (i > dimIndex && remove) {
            destSymmo->dimensionSize[i-1] = sourceSymmo->dimensionSize[i];
            continue;
          } 
          destSymmo->dimensionSize[i] = sourceSymmo->dimensionSize[i];
        }
        if (remove)
          destSymmo->shapeSize = ConstantExpr::create(shapeSize-1, Expr::Int64);
        else
          destSymmo->shapeSize = shapeSizeExpr;
      } else {
        destSymmo->dimensionSize = sourceSymmo->dimensionSize;
        destSymmo->strides = sourceSymmo->strides;
        destSymmo->shapeSize = sourceSymmo->shapeSize;
      }
      destSymmo->sizeName = sourceSymmo->sizeName;
      destSymmo->size = sourceSymmo->size;
      destSymmo->scalarType = sourceSymmo->scalarType;
      destSymmo->elementSize = sourceSymmo->elementSize;
      destSymmo->isQuantized = sourceSymmo->isQuantized;
      destSymmo->qscheme = sourceSymmo->qscheme;
      destSymmo->zeroPoint= sourceSymmo->zeroPoint;
      destSymmo->maxVal = sourceSymmo->maxVal;
      destSymmo->minVal = sourceSymmo->minVal;
      destSymmo->hasDuplicateVal = sourceSymmo->hasDuplicateVal;
      destSymmo->isFloat = sourceSymmo->isFloat;
    } else {
      terminateStateOnProgramError(state, "at4_ops11squeeze_dim4callERKNS_6TensorEl:dim index is not constant", StateTerminationType::ReportError);
        return;
    }
    return;
  }

  if (callable->getName().find("at6native12var_mean_outERNS_6TensorES2_RKS1_N3c108ArrayRefIlEElb") != std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "var_mean_out: do not find return tuple object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "var_mean_out: do not find out_var object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op2;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[2], op2, success) || !success) {
      terminateStateOnProgramError(state, "var_mean_out: do not find out_mean object", StateTerminationType::ReportError);
        return;
    }

    ObjectPair op3;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[3], op3, success) || !success) {
      terminateStateOnProgramError(state, "var_mean_out: do not find input tensor object", StateTerminationType::ReportError);
        return;
    }

    const MemoryObject *sourceMO = op3.first;
    std::string sourceSymName = getSymName(state, sourceMO, arguments[3]);
    const MemoryObject *destVarMO = op1.first;
    const MemoryObject *destMeanMO = op2.first;

    if (state.symbolicArrayMap.find(sourceSymName) == state.symbolicArrayMap.end()) {
      terminateStateOnProgramError(state, "var_mean_out: do not find source tensor symbolic object", StateTerminationType::ReportError);
      return;
    }

    SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[sourceSymName];
    SymArrayMemoryObject *destVarSymm = createSymArray(state, arguments[1], destVarMO, "var_mean_out", "var_tensor_");
    SymArrayMemoryObject *destMeanSymm = createSymArray(state, arguments[2], destMeanMO, "var_mean_out", "mean_tensor_");

    ObjectState* wos = state.addressSpace.getWriteable(op0.first, op0.second);
    wos->write(0, destVarMO->getBaseExpr());
    wos->write(8, destMeanMO->getBaseExpr());

    bool keepDim = false;
    if (auto keepExpr = dyn_cast<ConstantExpr>(arguments[7])) {
      if (keepExpr->isTrue()) {
        keepDim = true;
      }
    } else {
      terminateStateOnProgramError(state, "var_mean_out: keepDim is not constant", StateTerminationType::ReportError);
        return;
    }

    if (auto arraySizeExpr = dyn_cast<ConstantExpr>(arguments[5])) {
      uint64_t arraySize = arraySizeExpr->getZExtValue();
      ObjectPair op4;
      if (!state.addressSpace.resolveOne(state, solver.get(), arguments[4], op4, success) || !success) {
        terminateStateOnProgramError(state, "var_mean_out: do not find the dim array object", StateTerminationType::ReportError);
        return;
      }
      const ObjectState *arrayOS = op4.second;
      std::vector<int64_t> changeDims;
      for (uint64_t i = 0; i < arraySize; ++i) {
        ref<Expr> element = arrayOS->read(i*8, 64);

        if (auto CE = dyn_cast<ConstantExpr>(element)) {
          int64_t element_val = CE->getZExtValue();
          changeDims.emplace_back(element_val);
        }
      }
      if (keepDim) {
        destVarSymm->shapeSize = sourceSymmo->shapeSize;
        destMeanSymm->shapeSize = sourceSymmo->shapeSize;
      } else {
        ref<Expr> newShapeSize = SubExpr::create(ZExtExpr::create(sourceSymmo->shapeSize, Expr::Int64), arraySizeExpr);
        destVarSymm->shapeSize = newShapeSize;
        destMeanSymm->shapeSize = newShapeSize;
      }
      destVarSymm->scalarType = sourceSymmo->scalarType;
      destVarSymm->elementSize = sourceSymmo->elementSize;
      destVarSymm->isQuantized = sourceSymmo->isQuantized;
      destVarSymm->qscheme = sourceSymmo->qscheme;
      destVarSymm->zeroPoint= sourceSymmo->zeroPoint;

      destMeanSymm->scalarType = sourceSymmo->scalarType;
      destMeanSymm->elementSize = sourceSymmo->elementSize;
      destMeanSymm->isQuantized = sourceSymmo->isQuantized;
      destMeanSymm->qscheme = sourceSymmo->qscheme;
      destMeanSymm->zeroPoint= sourceSymmo->zeroPoint;
      
      if (!sourceSymmo->shapeSize.isNull() && isa<ConstantExpr>(sourceSymmo->shapeSize)) {
        auto shapeSizeExpr = dyn_cast<ConstantExpr>(sourceSymmo->shapeSize);
        uint64_t shapeSize = shapeSizeExpr->getZExtValue();
        int index = 0;
        ref<Expr> sizeExpr = ConstantExpr::create(1, 64);
        for (uint64_t i = 0; i < shapeSize; ++i) {
          if (std::find(changeDims.begin(), changeDims.end(), i)!=changeDims.end()) {
            if (keepDim) {
              destVarSymm->dimensionSize[index] = ConstantExpr::create(1, 64);
              destMeanSymm->dimensionSize[index] = ConstantExpr::create(1, 64);
            }
          } else {
            destVarSymm->dimensionSize[index] = sourceSymmo->dimensionSize[i];
            destMeanSymm->dimensionSize[index] = sourceSymmo->dimensionSize[i];
            sizeExpr = MulExpr::create(sizeExpr, ZExtExpr::create(sourceSymmo->dimensionSize[i], Expr::Int64));
          }
          index+=1;
        }
        destVarSymm->size = sizeExpr;
        destMeanSymm->size = sizeExpr;
      } else {
        destVarSymm->size = sourceSymmo->size;
        destVarSymm->dimensionSize = sourceSymmo->dimensionSize;
        destVarSymm->strides = sourceSymmo->strides;

        destMeanSymm->size = sourceSymmo->size;
        destMeanSymm->dimensionSize = sourceSymmo->dimensionSize;
        destMeanSymm->strides = sourceSymmo->strides;
      }
    } else {
      terminateStateOnProgramError(state, "var_mean_out: array size is not constant", StateTerminationType::ReportError);
      return;
    }
    return;
  }

  if (callable->getName().find("checkBackendEPKcN3c108ArrayRefINS_6TensorEEENS2_7Backend") != std::string::npos) {
    return;
  }

  if (callable->getName().find("3c1013MessageLoggerC1EPKcii") != std::string::npos) {
    return;
  }

  if (callable->getName().find("2at6native14_chunk_cat_outEN3c108ArrayRefINS_6TensorEEEllRS3_") != std::string::npos) { // TODO
    return;
  }

  if (callable->getName().find("tlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc")!=std::string::npos || callable->getName().find("ZNSolsEPFRSoS_E")!=std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("ZNSolsEd")!=std::string::npos || callable->getName().find("ZNSolsEi")!=std::string::npos) {
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("2at6native37foreach_tensor_copy_list_kernel_slow_EN3c108ArrayRefINS_6TensorEEES4_b") != std::string::npos) { // TODO
    return;
  }

  if (callable->getName().find("TensorIteratorConfig")!=std::string::npos && callable->getName().find("10TensorBase")!=std::string::npos 
      && callable->getName().find("add")!=std::string::npos && callable->getName().find("const_input")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "TensorIteratorConfig::add_const_input: do not find tensor object", StateTerminationType::ReportError);
      return;
    }

    const MemoryObject *mo = op.first;
    state.inputIterator->addTensor(mo->name, true);
    bindLocal(target, state, arguments[0]);
    return;
  } else if (callable->getName().find("TensorIteratorConfig")!=std::string::npos && callable->getName().find("10TensorBase")!=std::string::npos 
      && callable->getName().find("add")!=std::string::npos && callable->getName().find("_input")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "TensorIteratorConfig::add_const_input: do not find tensor object", StateTerminationType::ReportError);
      return;
    }

    const MemoryObject *mo = op.first;
    state.inputIterator->addTensor(mo->name, false);
    bindLocal(target, state, arguments[0]);
    return;
  } else if (callable->getName().find("TensorIteratorConfig")!=std::string::npos && callable->getName().find("10TensorBase")!=std::string::npos 
      && callable->getName().find("add")!=std::string::npos && callable->getName().find("const_output")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "TensorIteratorConfig::add_output: do not find tensor object", StateTerminationType::ReportError);
      return;
    }

    const MemoryObject *mo = op.first;
    state.inputIterator->addTensor(mo->name, true);
    state.inputIterator->noutputs+=1;
    bindLocal(target, state, arguments[0]);
    return;
  } else if (callable->getName().find("TensorIteratorConfig")!=std::string::npos && callable->getName().find("10TensorBase")!=std::string::npos 
      && callable->getName().find("add")!=std::string::npos && callable->getName().find("_output")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "TensorIteratorConfig::add_output: do not find tensor object", StateTerminationType::ReportError);
      return;
    }

    const MemoryObject *mo = op.first;
    state.inputIterator->addTensor(mo->name, false);
    state.inputIterator->noutputs+=1;
    bindLocal(target, state, arguments[0]);
    return;
  }

  if (callable->getName().find("cudaPeekAtLastError")!=std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("at14check_dim_sizeERKNS_6Tensor")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "check_dim_size: do not find tensor object", StateTerminationType::ReportError);
      return;
    }

    const MemoryObject *mo = op.first;
    std::string symName = getSymName(state, mo, arguments[0]);
    if (state.symbolicArrayMap.find(symName) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
      if (symmo->shapeSize.isNull()) {
        std::string dimName = symName + ".dim";
        auto dimpair = createSizeSymbol(state, dimName);
        ref<Expr> dimExpr = dimpair.second; 
        symmo->shapeSize = dimExpr;
      }
      ref<Expr> cond = EqExpr::create(ZExtExpr::create(symmo->shapeSize, Expr::Int64), arguments[1]);

      if (auto ce = dyn_cast<ConstantExpr>(arguments[2])) {
        int index = ce->getZExtValue();
        if (symmo->dimensionSize.find(index) == symmo->dimensionSize.end()) {
          std::string sizeName = symName + ".size[" + std::to_string(index)+"]";
          auto sizepair = createSizeSymbol(state, sizeName);
          ref<Expr> sizeExpr = sizepair.second; 
          addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));  
          symmo->dimensionSize[index] = sizeExpr;
        }
        cond  = AddExpr::create(cond, EqExpr::create(ZExtExpr::create(symmo->dimensionSize[index], Expr::Int64), arguments[3]));
      } else {
        terminateStateOnProgramError(state, "check_dim_size: dim index is not constant", StateTerminationType::ReportError);
        return;
      }
      StatePair branches = fork(state, cond, true, BranchType::Conditional);
      if (branches.second) {
        terminateStateOnProgramError(*branches.second, "check_dim_size: do not statisfy", StateTerminationType::Assert);
      }
    } else {
      terminateStateOnProgramError(state, "check_dim_size: do not find tensor symbolic object", StateTerminationType::ReportError);
    }
    return;
  }

  if (callable->getName().find("at9checkSizeEPKcRKNS_17TensorGeometryArgEll")!=std::string::npos) { // TODO
    return;
  }

  if (callable->getName().find("c10eqElRKNS_6SymIntE")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "c10eqElRKNS_6SymIntE: do not find symint object", StateTerminationType::ReportError);
      return;
    }

    const ObjectState *os = op.second;
    ref<Expr> value = os->read(0, Expr::Int64);
    if (!value.isNull()) {
      if (auto CE1 = dyn_cast<ConstantExpr>(value)) {
        if (auto CE0 = dyn_cast<ConstantExpr>(arguments[0])) {
          int64_t a = CE0->getZExtValue();
          int64_t b = CE1->getZExtValue();
          if (a==b) {
            bindLocal(target, state, ConstantExpr::create(1, Expr::Bool));
            return;
          }
        }
      }
    }
    bindLocal(target, state, ConstantExpr::create(0, Expr::Bool));
    return;
  }

  if (callable->getName().find("3c10dvERKNS_6SymIntEi")!=std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "3c10dvERKNS_6SymIntEi: do not find symint object", StateTerminationType::ReportError);
      return;
    }
    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "3c10dvERKNS_6SymIntEi: do not find symint object", StateTerminationType::ReportError);
      return;
    }

    const ObjectState *os1 = op1.second;
    ref<Expr> value = os1->read(0, Expr::Int64);
    if (!value.isNull()) {
      ref<Expr> res = SDivExpr::create(value, ZExtExpr::create(arguments[2], Expr::Int64));
      ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
      wos->write(0, res);
    }
    return;
  }

  if (callable->getName().find("3c10mlERKNS_6SymIntEi")!=std::string::npos) {
    bool success;
    ObjectPair op0;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op0, success) || !success) {
      terminateStateOnProgramError(state, "3c10mlERKNS_6SymIntEi: do not find symint object", StateTerminationType::ReportError);
      return;
    }
    ObjectPair op1;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op1, success) || !success) {
      terminateStateOnProgramError(state, "3c10mlERKNS_6SymIntEi: do not find symint object", StateTerminationType::ReportError);
      return;
    }

    const ObjectState *os1 = op1.second;
    ref<Expr> value = os1->read(0, Expr::Int64);
    if (!value.isNull()) {
      ref<Expr> res = MulExpr::create(value, ZExtExpr::create(arguments[2], Expr::Int64));
      ObjectState *wos = state.addressSpace.getWriteable(op0.first, op0.second);
      wos->write(0, res);
    }
    return;
  }

  if (callable->getName().find("__usAtomicCAS")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[0], op, success) || !success) {
      terminateStateOnProgramError(state, "__usAtomicCAS: do not find object", StateTerminationType::ReportError);
      return;
    }

    const ObjectState *os = op.second;
    ref<Expr> oldValue = os->read(0, Expr::Int16);
    ref<Expr> cond = EqExpr::create(oldValue, arguments[1]);
    StatePair branches = fork(state, cond, true, BranchType::Conditional);
    if (branches.second) {
      bindLocal(target, *branches.second, oldValue);
    }
    if (branches.first) {
      ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
      wos->write(0, arguments[2]);
      bindLocal(target, *branches.first, oldValue);
    }
    return;
  }

  if (callable->getName().find("get_sm_version_numv")!=std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(90, Expr::Int32));
    return;
  }

  if (callable->getName().find("get_device_attributell")!=std::string::npos) {
    ref<Expr> kindExpr = arguments[0];
    ref<Expr> value;
    if (auto kindCE = dyn_cast<ConstantExpr>(kindExpr)) {
      int kind = kindCE->getZExtValue();
      if (kind == 1) {
        value = ConstantExpr::create(1024, Expr::Int32);
      } else if (kind == 16) {
        value = ConstantExpr::create(80, Expr::Int32);
      } else if (kind == 97) {
        value = ConstantExpr::create(228*1024, Expr::Int32);
      } else {
        std::string symName = "device_attr_" + llvm::utostr(kind);
        MemoryObject *mo = memory->allocate(4, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);          
        value = symbolizeIndex(state, symName, mo, false);
      }
      bindLocal(target, state, value);
    } else {
      terminateStateOnProgramError(state, "cuda device attribute not constant",
                                         StateTerminationType::ReportError);
    }
    return;
  }

  if (callable->getName().find("cublasGetMathMode")!=std::string::npos) {
    bool success;
    ObjectPair op;
    if (!state.addressSpace.resolveOne(state, solver.get(), arguments[1], op, success) || !success) {
      terminateStateOnProgramError(state, "cublasGetMathMode: do not find math mode object", StateTerminationType::ReportError);
      return;
    }
    ref<ConstantExpr> mathMode = ConstantExpr::create(0, Expr::Int32);
    ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
    wos->write(0, mathMode);
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("cublasGemmEx")!=std::string::npos) {
    bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
    return;
  }

  if (callable->getName().find("sqrtf")!=std::string::npos) {
    if (auto valCE = dyn_cast<ConstantExpr>(arguments[0])) {
      float f = ConstantExpr::getFloatFromConstantExpr(valCE);
      float res = std::sqrt(f);
      uint32_t bits;
      std::memcpy(&bits, &res, sizeof(float));
      bindLocal(target, state, ConstantExpr::create(bits, Expr::Int32, true, true));
      return;
    } else {
      MemoryObject *mo =
        memory->allocate(4, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);       
      unsigned id = 0;
      std::string uniqueName = "sqrtf";
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = "sqrtf" + llvm::utostr(++id);
      }
      const Array *array = arrayCache.CreateArray(uniqueName, mo->size, true);
      bindObjectInState(state, mo, false, array);
      state.addSymbolic(mo, array);
      ref<Expr> resExpr = Expr::createTempRead(array, mo->size*8);     
      addConstraint(state, EqExpr::create(arguments[0], MulExpr::create(resExpr, resExpr)));
      bindLocal(target, state, resExpr);
      return;
    }
  }

  if (callable->getName().find("checkAllContiguousEPKcN3c108ArrayRefINS_9TensorArgEEE")!=std::string::npos) {
    return;
  }
  if (callable->getName().find("checkDeviceTypeEPKcN3c108ArrayRefINS_6TensorEEENS2_10DeviceTypeE")!=std::string::npos) {
    return;
  }
  if (callable->getName().find("checkAllSameGPUEPKcN3c108ArrayRefINS_9TensorArgEEE")!=std::string::npos) {
    return;
  }
  if (callable->getName().find("checkSameGPUEPKcRKNS_9TensorArgES4_")!=std::string::npos) {
    return;
  }

  // normal external function handling path
  // we could iterate through all the arguments first and determine the exact
  // size we need, but this is faster, and the memory usage isn't significant.
  size_t allocatedBytes = Expr::MaxWidth / 8 * (arguments.size() + 1);
  uint64_t *args = (uint64_t *)alloca(allocatedBytes);
  memset(args, 0, allocatedBytes);
  unsigned wordIndex = 2;
  for (auto &a : arguments) {
    if (ExternalCalls == ExternalCallPolicy::All ||
        ExternalCalls == ExternalCallPolicy::OverApprox) {
      a = optimizer.optimizeExpr(a, true);
      ref<ConstantExpr> cvalue = toConstant(
          state, a, "external call", ExternalCalls == ExternalCallPolicy::All);
      cvalue->toMemory(&args[wordIndex]);

      // If the argument points to a valid and writable object, concretise it
      // according to the selected policy
      if (ObjectPair op;
          cvalue->getWidth() == Context::get().getPointerWidth() &&
          state.addressSpace.resolveOne(cvalue, op) && !op.second->readOnly) {
        auto *os = state.addressSpace.getWriteable(op.first, op.second);
        os->flushToConcreteStore(*this, state,
                                 ExternalCalls == ExternalCallPolicy::All);
      }

      wordIndex += (cvalue->getWidth() + 63) / 64;
    } else {
      ref<Expr> arg = toUnique(state, a);
      if (ConstantExpr *ce = dyn_cast<ConstantExpr>(arg)) {
        // fp80 must be aligned to 16 according to the System V AMD 64 ABI
        if (ce->getWidth() == Expr::Fl80 && wordIndex & 0x01)
          wordIndex++;

        // XXX kick toMemory functions from here
        ce->toMemory(&args[wordIndex]);
        wordIndex += (ce->getWidth() + 63) / 64;
      } else {
        terminateStateOnExecError(state,
                                  "external call with symbolic argument: " +
                                      callable->getName());
        return;
      }
    }
  }

  // Prepare external memory for invoking the function

  //          "allocator too full, assumption that each object occupies its own "

  //   // average of pages needed for an external function call
    state.addressSpace.copyOutConcretes();

#ifndef WINDOWS
  // Update external errno state with local state value
  int *errno_addr = getErrnoLocation(state);
  ObjectPair result;
  bool resolved = state.addressSpace.resolveOne(
      ConstantExpr::create((uint64_t)errno_addr, Expr::Int64), result);
  if (!resolved)
    klee_error("Could not resolve memory object for errno");
  ref<Expr> errValueExpr = result.second->read(0, sizeof(*errno_addr) * 8);
  ConstantExpr *errnoValue = dyn_cast<ConstantExpr>(errValueExpr);
  if (!errnoValue) {
    terminateStateOnExecError(state,
                              "external call with errno value symbolic: " +
                                  callable->getName());
    return;
  }

  externalDispatcher->setLastErrno(
      errnoValue->getZExtValue(sizeof(*errno_addr) * 8));
#endif

  if (ExternalCallWarnings != ExtCallWarnings::None) {
    std::string TmpStr;
    llvm::raw_string_ostream os(TmpStr);
    os << "calling external: " << callable->getName().str() << "(";
    for (unsigned i = 0; i < arguments.size(); i++) {
      os << arguments[i];
      if (i != arguments.size() - 1)
        os << ", ";
    }
    os << ") at " << state.pc->getSourceLocation();

    if (ExternalCallWarnings == ExtCallWarnings::All)
      klee_warning("%s", os.str().c_str());
    else
      klee_warning_once(callable->getValue(), "%s", os.str().c_str());
  }

  const bool isBasicStringInsert =
      callable->getName().find("NSt7__cxx1112basic_string") !=
          std::string::npos &&
      callable->getName().find("IcSt11char_traitsIcESaIcEE6insert") !=
          std::string::npos;
  if (isBasicStringInsert) {
    Type *resultType = target->inst->getType();
    if (!resultType->isVoidTy())
      bindLocal(target, state, arguments[0]);
    return;
  }

  bool success = externalDispatcher->executeCall(callable, target->inst, args);
  if (!success) {
    terminateStateOnExecError(state,
                              "failed external call: " + callable->getName(),
                              StateTerminationType::External);
    return;
  }

  if (!state.addressSpace.copyInConcretes(ExternalCalls ==
                                          ExternalCallPolicy::All)) {
    terminateStateOnExecError(state, "external modified read-only object",
                              StateTerminationType::External);
    return;
  }


#ifndef WINDOWS
  // Update errno memory object with the errno value from the call
  int error = externalDispatcher->getLastErrno();
  state.addressSpace.copyInConcrete(result.first, result.second,
                                    (uint64_t)&error,
                                    ExternalCalls == ExternalCallPolicy::All);
#endif

  Type *resultType = target->inst->getType();
  if (resultType != Type::getVoidTy(kmodule->module->getContext())) {
    ref<Expr> e =
        ConstantExpr::fromMemory((void *)args, getWidthForLLVMType(resultType));
    bindLocal(target, state, e);
  }
}

/***/

ref<Expr> Executor::replaceReadWithSymbolic(ExecutionState &state, 
                                            ref<Expr> e) {
  unsigned n = interpreterOpts.MakeConcreteSymbolic;
  if (!n || replayKTest || replayPath)
    return e;

  if (!isa<ConstantExpr>(e))
    return e;

  if (n != 1 && random() % n)
    return e;

  // create a new fresh location, assert it is equal to concrete value in e
  // and return it.
  
  static unsigned id;
  const Array *array =
      arrayCache.CreateArray("rrws_arr" + llvm::utostr(++id),
                             Expr::getMinBytesForWidth(e->getWidth()));
  ref<Expr> res = Expr::createTempRead(array, e->getWidth());
  ref<Expr> eq = NotOptimizedExpr::create(EqExpr::create(e, res));
  state.addConstraint(eq);
  return res;
}

ObjectState *Executor::bindObjectInState(ExecutionState &state, 
                                         const MemoryObject *mo,
                                         bool isLocal,
                                         const Array *array) {
  ObjectState *os = array ? new ObjectState(mo, array) : new ObjectState(mo);
  state.addressSpace.bindObject(mo, os);

  // Its possible that multiple bindings of the same mo in the state
  // will put multiple copies on this list, but it doesn't really
  // matter because all we use this list for is to unbind the object
  // on function return.
  if (isLocal)
    state.stack.back().allocas.push_back(mo);

  return os;
}

std::pair<const Array *, ref<Expr>> Executor::createSizeSymbol(ExecutionState &state, const std::string &sizeName, bool isTotal) {
  MemoryObject *sizeMo =
        memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                            &state, /*allocSite=*/state.prevPC->inst,
                            /*alignment=*/8);          
  sizeMo->setName(sizeName);
  const Array *sizeArray = arrayCache.CreateArray(sizeName, sizeMo->size, true);
  bindObjectInState(state, sizeMo, false, sizeArray);
  state.addSymbolic(sizeMo, sizeArray);
  ref<Expr> sizeExpr = Expr::createTempRead(sizeArray, sizeMo->size*8);

  addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(0, sizeExpr->getWidth())));  
  if (isTotal) {
    ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << (sizeExpr->getWidth()-1)) - 1, sizeExpr->getWidth());  
    addConstraint(state, UleExpr::create(sizeExpr, maxValueExpr));
  } else {
    if (sizeExpr->getWidth() < 32) {
      ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << sizeExpr->getWidth()) - 1, sizeExpr->getWidth());  
      addConstraint(state, UleExpr::create(sizeExpr, maxValueExpr));
    } else {
      ref<ConstantExpr> maxValueExpr = ConstantExpr::alloc((1ULL << 31) - 1, sizeExpr->getWidth());
      addConstraint(state, UleExpr::create(sizeExpr, maxValueExpr));
    }
  }
 
  return std::make_pair(sizeArray, sizeExpr);
}

ref<Expr> Executor::symbolizeIndex(ExecutionState &state, const std::string &name, const MemoryObject *mo, bool isSigned, ref<Expr> initialVal) { // to reduce search space in z3, assume the integers will not be larger than INT_MAX
  mo->setName(name);
  const Array *array = arrayCache.CreateArray(name, mo->size, true);
  ref<Expr> curVal;

  if (!initialVal.isNull()) {
    const ObjectState *osOld = state.addressSpace.findObject(mo);
    if (osOld) {
      curVal = osOld->read(0, mo->size * 8);
    }
  }

  ObjectState *os = bindObjectInState(state, mo, false, array);
  ref<Expr> indexExpr = Expr::createTempRead(array, mo->size * 8);
  state.addSymbolic(mo, array);
  
  if (!initialVal.isNull() && !curVal.isNull() && curVal != initialVal) {
    std::map< ref<Expr>, ref<Expr> > replacements;
    replacements[initialVal] = indexExpr;
    ref<Expr> newVal = ConstraintManager::replaceReadExpr(curVal, replacements);
    os->write(0, newVal);
  } else 
    os->write(0, indexExpr);

  if (isSigned) {
      ref<ConstantExpr> minValueExpr = ConstantExpr::alloc(1ULL << (indexExpr->getWidth()-1), indexExpr->getWidth(), true);   
      addConstraint(state, SgeExpr::create(indexExpr, minValueExpr));  
      ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << (indexExpr->getWidth()-1)) - 1, indexExpr->getWidth());
      addConstraint(state, UleExpr::create(indexExpr, maxValueExpr));
  } else {
    addConstraint(state, UgeExpr::create(indexExpr, ConstantExpr::create(0, indexExpr->getWidth())));  
    if (indexExpr->getWidth() < 64) {
      ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << indexExpr->getWidth()) - 1, indexExpr->getWidth()); 
      addConstraint(state, UleExpr::create(indexExpr, maxValueExpr));     
    } else if (indexExpr->getWidth() == 64) {
      ref<ConstantExpr> maxValueExpr = ConstantExpr::alloc(18446744073709551615ULL, 64);
      addConstraint(state, UleExpr::create(indexExpr, maxValueExpr));
    }
  }

  return indexExpr;
}

ref<Expr> Executor::createIndexRangeExpr(ExecutionState &state, const std::string &name, uint64_t size, bool isLocal, ref<Expr> rangeMin, ref<Expr> rangeMax, bool compareSigned, const MemoryObject *mo) {
    if (!mo)
      mo = memory->allocate(size, isLocal, /*isGlobal=*/false,
                         &state, nullptr, 4);
    mo->setName(name);
    const Array *array = arrayCache.CreateArray(name, mo->size, true);
    bindObjectInState(state, mo, isLocal, array);

    ref<Expr> indexExpr = Expr::createTempRead(array, mo->size * 8);
    state.addSymbolic(mo, array);

    if (rangeMin) {
      ref<Expr> lowerBound;
      if (compareSigned) {
        lowerBound = SgeExpr::create(indexExpr, rangeMin);
      } else {
        lowerBound = UgeExpr::create(indexExpr, rangeMin);
      }
      addConstraint(state, lowerBound);
    } else {
      if (compareSigned && indexExpr->getWidth() <= 64) {
        ref<ConstantExpr> minValueExpr = ConstantExpr::alloc(1ULL << (indexExpr->getWidth()-1), indexExpr->getWidth(), true);   
        addConstraint(state, SgeExpr::create(indexExpr, minValueExpr));  
      } else if (!compareSigned)
        addConstraint(state, UgeExpr::create(indexExpr, ConstantExpr::create(0, indexExpr->getWidth())));  
    }

    if (rangeMax){
      ref<Expr> upperBound;
      if (compareSigned) {
        upperBound = SltExpr::create(indexExpr, rangeMax);
      } else {
        upperBound = UltExpr::create(indexExpr, rangeMax);
      }
      addConstraint(state, upperBound);  
    } else {
      if (indexExpr->getWidth() <= 64) {
        if (compareSigned){
          ref<ConstantExpr> maxValueExpr = ConstantExpr::create((1ULL << (indexExpr->getWidth()-1)) - 1, indexExpr->getWidth()); 
          addConstraint(state, UleExpr::create(indexExpr, maxValueExpr));  
        } else {
          ref<ConstantExpr> maxValueExpr;
          if (indexExpr->getWidth() == 64) {
            maxValueExpr = ConstantExpr::create((1ULL << (indexExpr->getWidth()-1)) - 1, indexExpr->getWidth()); 
          } else {
            maxValueExpr = ConstantExpr::create((1ULL << 63) - 1, indexExpr->getWidth()); 
          }
          addConstraint(state, UleExpr::create(indexExpr, maxValueExpr));
        }     
      }
    }
    return indexExpr;
}

void Executor::validateKernelConfig(ExecutionState &state, ref<Expr> config, ref<Expr> maxValueExpr) {
  if (maxValueExpr.isNull()) {
    if (config->getWidth() < 32) {
      maxValueExpr = ConstantExpr::create((1ULL << config->getWidth()) - 1, config->getWidth());  
    } else {
      maxValueExpr = ConstantExpr::create((1ULL << 31) - 1, config->getWidth());  
    }
  }

  ref<Expr> con = SgeExpr::create(config, ConstantExpr::create(0, config->getWidth()));
  con = AndExpr::create(con, UleExpr::create(config, maxValueExpr));
  StatePair branches = fork(state, con, true, BranchType::Conditional);
  if (branches.second) {
    terminateStateOnProgramError(*branches.second, "kernel config error", StateTerminationType::Overflow);
  }
}

void Executor::initializeCudaKernelConfig(ExecutionState &state, ref<Expr> gridxExpr, ref<Expr> gridyExpr, ref<Expr> gridzExpr, 
                                          ref<Expr> blockxExpr, ref<Expr> blockyExpr, ref<Expr> blockzExpr, ref<Expr> sharedMemSizeExpr){
  CUDAKernelConfig::Dim3 grid(gridxExpr, gridyExpr);
  if (gridzExpr) {
    grid.z = gridzExpr;
    if (!isa<ConstantExpr>(gridzExpr)) {
      ref<Expr> maxValueExpr = ConstantExpr::create(65536, blockzExpr->getWidth());  
      validateKernelConfig(state, gridzExpr, maxValueExpr);
    }
  }
  if (!isa<ConstantExpr>(gridxExpr)) {
    validateKernelConfig(state, gridxExpr);
  }
  if (!isa<ConstantExpr>(gridyExpr)) {
    ref<Expr> maxValueExpr = ConstantExpr::create(65536, blockzExpr->getWidth()); 
    validateKernelConfig(state, gridyExpr, maxValueExpr);
  }
  CUDAKernelConfig::Dim3 block(blockxExpr, blockyExpr);
  if (blockzExpr) {
    block.z = blockzExpr;
    if (!isa<ConstantExpr>(blockzExpr)) {
      ref<Expr> maxValueExpr = ConstantExpr::create(64, blockzExpr->getWidth());  
      validateKernelConfig(state, blockzExpr, maxValueExpr);
    }
  }
  if (!isa<ConstantExpr>(blockxExpr)) {
    ref<Expr> maxValueExpr = ConstantExpr::create(1024, blockxExpr->getWidth());  
    validateKernelConfig(state, blockxExpr, maxValueExpr);
  }
  if (!isa<ConstantExpr>(blockyExpr)) {
    ref<Expr> maxValueExpr = ConstantExpr::create(1024, blockyExpr->getWidth());  
    validateKernelConfig(state, blockyExpr, maxValueExpr);
  }
  CUDAKernelConfig kernelConfig;
  kernelConfig.setGridDim(grid);
  kernelConfig.setBlockDim(block);
  kernelConfig.setSharedMemSize(sharedMemSizeExpr);

  ref<Expr> blockXIndex = createIndexRangeExpr(state, "blockIdx.x"+llvm::utostr(kernelConfigCounter), sizeof(unsigned), false, ConstantExpr::create(0, sizeof(unsigned)*8), gridxExpr);
  ref<Expr> blockYIndex = createIndexRangeExpr(state, "blockIdx.y"+llvm::utostr(kernelConfigCounter), sizeof(unsigned), false, ConstantExpr::create(0, sizeof(unsigned)*8), gridyExpr);
  ref<Expr> blockZIndex;
  if (gridzExpr)
    blockZIndex = createIndexRangeExpr(state, "blockIdx.z"+llvm::utostr(kernelConfigCounter), sizeof(unsigned), false, ConstantExpr::create(0, sizeof(unsigned)*8), gridzExpr);
  else 
    blockZIndex = ConstantExpr::create(0, Expr::Int32);
  kernelConfig.setBlockIdx(CUDAKernelConfig::Dim3(blockXIndex, blockYIndex, blockZIndex));

  ref<Expr> threadXIndex = createIndexRangeExpr(state, "threadIdx.x"+llvm::utostr(kernelConfigCounter), sizeof(unsigned), false, ConstantExpr::create(0, sizeof(unsigned)*8), blockxExpr);
  ref<Expr> threadYIndex = createIndexRangeExpr(state, "threadIdx.y"+llvm::utostr(kernelConfigCounter), sizeof(unsigned), false, ConstantExpr::create(0, sizeof(unsigned)*8), blockyExpr);
  ref<Expr> threadZIndex;
  if (blockzExpr)
    threadZIndex = createIndexRangeExpr(state, "threadIdx.z"+llvm::utostr(kernelConfigCounter), sizeof(unsigned), false, ConstantExpr::create(0, sizeof(unsigned)*8), blockzExpr);
  else 
    threadZIndex = ConstantExpr::create(0, Expr::Int32);
  kernelConfig.setThreadIdx(CUDAKernelConfig::Dim3(threadXIndex, threadYIndex, threadZIndex));

  kernelConfigCounter++;
  state.setCudaKernelConfig(kernelConfig);
}

void Executor::executeAlloc(ExecutionState &state,
                            ref<Expr> size,
                            bool isLocal,
                            KInstruction *target,
                            bool zeroMemory,
                            const ObjectState *reallocFrom,
                            size_t allocationAlignment, ref<Expr> cudaAddress) {
  size = toUnique(state, size);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(size)) {
    const llvm::Value *allocSite = state.prevPC->inst;
    if (allocationAlignment == 0) {
      allocationAlignment = getAllocationAlignment(allocSite);
    }
    uint64_t sizeVal = std::max(CE->getZExtValue(), uint64_t(4));

    bool isGlobal = false;
    if (cudaAddress) {
      isGlobal = true;
    }
    MemoryObject *mo =
        memory->allocate(sizeVal, isLocal, /*isGlobal=*/isGlobal,
                         &state, allocSite, allocationAlignment);
    if (!mo) {
      bindLocal(target, state, 
                ConstantExpr::create(1, Context::get().getPointerWidth()));
    } else {
      auto it = std::find(deletedAddresses.begin(), deletedAddresses.end(), mo->address);
      if (it != deletedAddresses.end()) {
        deletedAddresses.erase(it);
      }

      ObjectState *os = bindObjectInState(state, mo, isLocal);
      if (zeroMemory) {
        os->initializeToZero();
      } else {
        os->initializeToRandom();
      }
      
      if (cudaAddress){
        ObjectPair op;
        bool success;
        
        if (state.addressSpace.resolveOne(state, solver.get(), cudaAddress, op, success) && success) {
          if (op.second->readOnly) {
            terminateStateOnProgramError(state, "memory error: object read only",
                                         StateTerminationType::ReadOnly);
            return;
          } else {
            ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
            wos->write(0, mo->getBaseExpr());
            bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
          }
        } else {
          klee_warning("cudaMalloc address not found");
          bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
        }
      } else{
        bindLocal(target, state, mo->getBaseExpr());
      }
      
      if (reallocFrom) {
        unsigned count = std::min(reallocFrom->size, os->size);
        for (unsigned i=0; i<count; i++)
          os->write(i, reallocFrom->read8(i));
        const MemoryObject *reallocObject = reallocFrom->getObject();
        state.deallocate(reallocObject);
        state.addressSpace.unbindObject(reallocObject);
      }
    }
  } else {
    // XXX For now we just pick a size. Ideally we would support
    // symbolic sizes fully but even if we don't it would be better to
    // "smartly" pick a value, for example we could fork and pick the
    // 
    // It would also be nice to recognize the case when size has
    // collapses the size expression with a select.

    size = optimizer.optimizeExpr(size, true);
    unsigned id = 0;
    std::string symName = "_alloc_0";
    while (!state.arrayNames.insert(symName).second) {
      symName = "_alloc_" + llvm::utostr(++id);
    }

    MemoryObject *mo =
          memory->allocate(largeSymbolSize, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);    
    if (!mo)
      klee_error("Could not allocate memory");

    mo->setName(symName);
    const Array *array = arrayCache.CreateArray(symName, mo->size);
    bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);

    auto pair = createSizeSymbol(state, symName + ".size");
    ref<Expr> sizeExpr = pair.second;

    SymArrayMemoryObject *symmo = new SymArrayMemoryObject(mo->address, array, pair.first, sizeExpr);
    symmo->shapeSize = ConstantExpr::create(1, sizeExpr->getWidth());
    state.symbolicArrayMap[symName] = symmo;

    ObjectState *os = bindObjectInState(state, mo, isLocal);
    if (zeroMemory) {
      os->initializeToZero();
    } else {
      os->initializeToRandom();
    }
    
    if (cudaAddress) {
      ObjectPair op;
      bool success;
      
      if (state.addressSpace.resolveOne(state, solver.get(), cudaAddress, op, success) && success) {
        if (op.second->readOnly) {
          terminateStateOnProgramError(state, "memory error: object read only",
                                        StateTerminationType::ReadOnly);
          return;
        } else {
          ObjectState *wos = state.addressSpace.getWriteable(op.first, op.second);
          wos->write(0, mo->getBaseExpr());
          bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
        }
      } else {
        bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
      }
    } else{
      bindLocal(target, state, mo->getBaseExpr());
    }

    // Check if in seed mode, then try to replicate size from a seed

  //     // Try and start with a small example.


  //     // Check for exactly two values
  //       // See if a *really* big value is possible. If so assume
  //       // malloc will fail for it, so lets fork and return 0.
        

  //                                      "concretized symbolic size",

  }
}

void Executor::executeFree(ExecutionState &state,
                           ref<Expr> address,
                           KInstruction *target) {
  address = optimizer.optimizeExpr(address, true);
  StatePair zeroPointer =
      fork(state, Expr::createIsZero(address), true, BranchType::Free);
  if (zeroPointer.first) {
    if (target)
      bindLocal(target, *zeroPointer.first, Expr::createPointer(0));
  }
  if (zeroPointer.second) { // address != 0
    ExactResolutionList rl;
    resolveExact(*zeroPointer.second, address, rl, "free");
    
    for (Executor::ExactResolutionList::iterator it = rl.begin(), 
           ie = rl.end(); it != ie; ++it) {
      const MemoryObject *mo = it->first.first;
      if (mo->isLocal) {
        terminateStateOnProgramError(*it->second, "free of alloca",
                                     StateTerminationType::Free,
                                     getAddressInfo(*it->second, address));
      } else if (mo->isGlobal) {
        terminateStateOnProgramError(*it->second, "free of global",
                                     StateTerminationType::Free,
                                     getAddressInfo(*it->second, address));
      } else {
        it->second->deallocate(mo);
        it->second->addressSpace.unbindObject(mo);
        if (target)
          bindLocal(target, *it->second, Expr::createPointer(0));
      }
    }
  }
}

void Executor::resolveExact(ExecutionState &state,
                            ref<Expr> p,
                            ExactResolutionList &results, 
                            const std::string &name) {
  p = optimizer.optimizeExpr(p, true);
  // XXX we may want to be capping this?
  ResolutionList rl;
  state.addressSpace.resolve(state, solver.get(), p, rl);
  
  ExecutionState *unbound = &state;
  for (ResolutionList::iterator it = rl.begin(), ie = rl.end(); 
       it != ie; ++it) {
    ref<Expr> inBounds = EqExpr::create(p, it->first->getBaseExpr());

    StatePair branches =
        fork(*unbound, inBounds, true, BranchType::ResolvePointer);

    if (branches.first)
      results.push_back(std::make_pair(*it, branches.first));

    unbound = branches.second;
    if (!unbound) // Fork failure
      break;
  }

  if (unbound) {
    auto CE = dyn_cast<ConstantExpr>(p);
    if (MemoryManager::isDeterministic && CE) {
      using kdalloc::LocationInfo;
      auto ptr = reinterpret_cast<void *>(CE->getZExtValue());
      auto li = unbound->heapAllocator.locationInfo(ptr, 1);
      if (li == LocationInfo::LI_AllocatedOrQuarantined &&
          li.getBaseAddress() == ptr && name == "free") {
        terminateStateOnProgramError(*unbound, "memory error: double free",
                                     StateTerminationType::Ptr,
                                     getAddressInfo(*unbound, p));
        return;
      }
    }
    terminateStateOnProgramError(
        *unbound, "memory error: invalid pointer: " + name,
        StateTerminationType::Ptr, getAddressInfo(*unbound, p));
  }
}

void Executor::executeCudaMemcpy(ExecutionState &state,
                                      ref<Expr> desAddress,
                                      ref<Expr> srcAddress,
                                      ref<Expr> size,
                                      KInstruction *target) {
  unsigned bytes;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(size)) {
    bytes = CE->getZExtValue();
  } else { // for symoblic size, directly check bound but do not read value and write value
    klee_warning("cuda memcpy size not constant");

    ObjectPair desOp;
    ObjectPair srcOp;
    bool desSuccess;
    bool srcSuccess;
    solver->setTimeout(coreSolverTimeout);

    if (!isa<ConstantExpr>(srcAddress)) {
      auto base_it = state.base_addrs.find(srcAddress);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, srcOp,
                                          srcSuccess) ||
            !srcSuccess) {
          klee_warning("Failed to resolve concrete address from the base_addrs "
                      "map to a memory object");
        }
      }
    }
    if (!srcSuccess && isa<ConstantExpr>(srcAddress)) {
      state.addressSpace.resolveOne(state, solver.get(), srcAddress, srcOp, srcSuccess);
    }

    if (!srcSuccess) {
      bindLocal(target, state, ConstantExpr::create(0, Context::get().getPointerWidth()));
      return;
    }

    const MemoryObject *srcMo = srcOp.first;
    ref<Expr> srcOffset = srcMo->getOffsetExpr(srcAddress);
    std::string srcSymName = getSymName(state, srcMo, srcAddress);
    if (state.symbolicArrayMap.find(srcSymName)!=state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[srcSymName];
      srcOffset = SubExpr::create(srcAddress, ConstantExpr::createPointer(symmo->arrayAddress));
      ref<Expr> sizeConstraint;
      setElementSize(state, symmo);
      if (symmo->elementSize) {
        ref<Expr> sizeBytes = MulExpr::create(symmo->size, ZExtExpr::create(symmo->elementSize, symmo->size->getWidth()));
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(srcOffset, sizeBytes, size);
      } else {
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(srcOffset, symmo->size, size);
      }
      sizeConstraint = AndExpr::create(sizeConstraint, SgeExpr::create(srcOffset, ConstantExpr::create(0, srcOffset->getWidth())));

      StatePair branches = fork(state, sizeConstraint, true, BranchType::MemOp);
      ExecutionState *unbound = branches.second;
      if (unbound) {
        terminateStateOnProgramError(*unbound, "out of bound pointer", StateTerminationType::Ptr);
      }
    }

    if (!isa<ConstantExpr>(desAddress)) {
      auto base_it = state.base_addrs.find(desAddress);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, desOp,
                                          desSuccess) ||
            !desSuccess) {
          klee_warning("Failed to resolve concrete address from the base_addrs "
                      "map to a memory object");
        }
      }
    }
    if (!desSuccess && isa<ConstantExpr>(desAddress)) {
      state.addressSpace.resolveOne(state, solver.get(), desAddress, desOp, desSuccess);
    }

    if (!desSuccess) {
      bindLocal(target, state, ConstantExpr::create(0, Context::get().getPointerWidth()));
      return;
    }

    const MemoryObject *desMo = desOp.first;
    ref<Expr> desOffset = desMo->getOffsetExpr(desAddress);
    std::string destSymName = getSymName(state, desMo, desAddress);
    if (state.symbolicArrayMap.find(destSymName)!=state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[destSymName];
      desOffset = SubExpr::create(desAddress, ConstantExpr::createPointer(symmo->arrayAddress));
      ref<Expr> sizeConstraint;
      setElementSize(state, symmo);
      if (symmo->elementSize) {
        ref<Expr> sizeBytes = MulExpr::create(symmo->size, ZExtExpr::create(symmo->elementSize, symmo->size->getWidth()));
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(desOffset, sizeBytes, size);
      } else {
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(desOffset, symmo->size, size);
      }
      sizeConstraint = AndExpr::create(sizeConstraint, SgeExpr::create(desOffset, ConstantExpr::create(0, desOffset->getWidth())));

      StatePair branches = fork(state, sizeConstraint, true, BranchType::MemOp);
      ExecutionState *unbound = branches.second;
      if (unbound) {
        terminateStateOnProgramError(*unbound, "out of bound pointer", StateTerminationType::Ptr);
      }
    }
    bindLocal(target, state, ConstantExpr::create(0, Context::get().getPointerWidth()));
    return;
  }
  srcAddress = optimizer.optimizeExpr(srcAddress, true);
  
  ObjectPair desOp;
  ObjectPair srcOp;
  bool desSuccess;
  bool srcSuccess;
  solver->setTimeout(coreSolverTimeout);

  bool resolveSingleObject = SingleObjectResolution;

  if (resolveSingleObject && !isa<ConstantExpr>(srcAddress)) {
    // Address is symbolic

    resolveSingleObject = false;
    auto base_it = state.base_addrs.find(srcAddress);
    if (base_it != state.base_addrs.end()) {
      // Concrete address found in the map, now find the associated memory
      // object
      if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, srcOp,
                                         srcSuccess) ||
          !srcSuccess) {
        klee_warning("Failed to resolve concrete address from the base_addrs "
                     "map to a memory object");
      } else {
        // We have resolved the stored concrete address to a memory object.
        // Now let's see if we can prove an overflow - we are only interested in
        // two cases: either we overflow and it's a bug or we don't and we carry
        // objects
        resolveSingleObject = true;
      }
    }
  } else {
    resolveSingleObject = false;
  }

  if (!resolveSingleObject) {
    if (!state.addressSpace.resolveOne(state, solver.get(), srcAddress, srcOp, srcSuccess)) {
      srcAddress = toConstant(state, srcAddress, "resolveOne failure");
      srcSuccess = state.addressSpace.resolveOne(cast<ConstantExpr>(srcAddress), srcOp);
    }
  }

  if (!isa<ConstantExpr>(desAddress)) {
    resolveSingleObject = false;
    auto base_it = state.base_addrs.find(desAddress);
    if (base_it != state.base_addrs.end()) {
      // Concrete address found in the map, now find the associated memory object
      if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, desOp,
                                         desSuccess) ||
          !desSuccess) {
        klee_warning("Failed to resolve concrete address from the base_addrs "
                     "map to a memory object");
      } else {
        // We have resolved the stored concrete address to a memory object.
        // Now let's see if we can prove an overflow - we are only interested in
        // two cases: either we overflow and it's a bug or we don't and we carry
        // objects
        resolveSingleObject = true;
      }
    }
  } else {
    resolveSingleObject = false;
  }

  if (!resolveSingleObject) {
    if (!state.addressSpace.resolveOne(state, solver.get(), desAddress, desOp, desSuccess)) {
      desAddress = toConstant(state, desAddress, "resolveOne failure");
      desSuccess = state.addressSpace.resolveOne(cast<ConstantExpr>(desAddress), desOp);
    }
  }
  solver->setTimeout(time::Span());

  if (srcSuccess && desSuccess) {
    const MemoryObject *mo = srcOp.first;

    if (MaxSymArraySize && mo->size >= MaxSymArraySize) {
      srcAddress = toConstant(state, srcAddress, "max-sym-array-size");
    }

    ref<Expr> offset = mo->getOffsetExpr(srcAddress);
    ref<Expr> check = mo->getBoundsCheckOffset(offset, bytes);
    check = AndExpr::create(check, SgeExpr::create(offset, ConstantExpr::create(0, offset->getWidth())));

    if (state.symbolicArrayMap.find(mo->name)!=state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *symmo = state.symbolicArrayMap[mo->name];
      ref<Expr> sizeConstraint;
      setElementSize(state, symmo);
      if (symmo->elementSize) {
        ref<Expr> sizeBytes = MulExpr::create(symmo->size, ZExtExpr::create(symmo->elementSize, symmo->size->getWidth()));
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(offset, sizeBytes, bytes);
      } else {
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(offset, symmo->size, bytes);
      }
      sizeConstraint = AndExpr::create(sizeConstraint, SgeExpr::create(offset, ConstantExpr::create(0, offset->getWidth())));

      StatePair branches = fork(state, sizeConstraint, true, BranchType::MemOp);
      ExecutionState *bound = branches.first;
      const ObjectState *os = srcOp.second;

      if (bound) {
        ObjectState* new_os = os->cloneFor(desOp.first, *desOp.second);             
        state.addressSpace.bindObject(desOp.first, new_os);
        bindLocal(target, state, ConstantExpr::create(0, Context::get().getPointerWidth()));
      }
      
      ExecutionState *unbound = branches.second;
      if (unbound) {
        terminateStateOnProgramError(*unbound, "out of bound pointer", StateTerminationType::Ptr);
      }
      return;
    }
    check = optimizer.optimizeExpr(check, true);

    bool inBounds;
    solver->setTimeout(coreSolverTimeout);
    bool success = solver->mustBeTrue(state.constraints, check, inBounds,
                                      state.queryMetaData, state.intArrNames, state.getIOExprs());
    solver->setTimeout(time::Span());
    if (!success) {
      state.pc = state.prevPC;
      terminateStateOnSolverError(state, "Query timed out (bounds check).");
      return;
    }

    if (inBounds) {
      const ObjectState *os = srcOp.second;
      ObjectState* new_os = os->cloneFor(desOp.first, *desOp.second);             
      state.addressSpace.bindObject(desOp.first, new_os);
      bindLocal(target, state, ConstantExpr::create(0, Context::get().getPointerWidth()));
      return;
    } else if (auto checkce = dyn_cast<ConstantExpr>(check)) {
      if (checkce->isZero()) {
        // llvm::outs() << "srcAddress " << srcAddress << " mo->address: " << mo->address << " offset " << offset << " bytes " << bytes << " size " << mo->size << "\n";
        terminateStateOnProgramError(state, "cannot handle memory operation", StateTerminationType::ReportError);
        return;
      }
    }
  }
}

bool Executor::exprContainsConstant(ref<Expr> expr, uint64_t target) {
  if (auto *ce = llvm::dyn_cast<ConstantExpr>(expr)) {
      return ce->getZExtValue() == target;
  }

  for (unsigned i = 0; i < expr->getNumKids(); ++i) {
      if (exprContainsConstant(expr->getKid(i), target)) {
          return true;
      }
  }

  return false;
}

void Executor::checkDataRace(ExecutionState &state, KLoopInfo *loopInfo) {
  if (state.addressInLoop.find(loopInfo->indexName) == state.addressInLoop.end()) {
    return;
  }
  if (state.addressInLoop[loopInfo->indexName].empty()) {
    return;
  }
  if (!state.kernelConfig.getThreadIdx().x) {
    return;
  }
  ConstraintSet globalBaseConstraints(state.constraints);
  ref<Expr> allGlobalNewCon;
  std::map< ref<Expr>, ref<Expr> > globalReplacements;
  ref<Expr> globalThreadCon;

  ConstraintSet sharedBaseConstraints(state.constraints);
  ref<Expr> allSharedNewCon;
  std::map< ref<Expr>, ref<Expr> > sharedReplacements;
  ref<Expr> sharedblockCon;

  ref<Expr> blockX = state.kernelConfig.getBlockIdx().x;
  ref<ReadExpr> tempRead = ReadExpr::extractReadExpr(blockX);
  const Array *newBXArray = arrayCache.CreateArray(tempRead->updates.root->name+"_2", sizeof(unsigned), true);
  ref<Expr> newblockX = Expr::createTempRead(newBXArray, sizeof(unsigned) * 8);
  globalReplacements[blockX] = newblockX;
  globalThreadCon = EqExpr::create(blockX, newblockX);

  ref<Expr> blockY = state.kernelConfig.getBlockIdx().y;
  tempRead = ReadExpr::extractReadExpr(blockY);
  const Array *newBYArray = arrayCache.CreateArray(tempRead->updates.root->name+"_2", sizeof(unsigned), true);
  ref<Expr> newblockY = Expr::createTempRead(newBYArray, sizeof(unsigned) * 8);
  globalReplacements[blockY] = newblockY;
  globalThreadCon = AndExpr::create(globalThreadCon, EqExpr::create(blockY, newblockY));

  ref<Expr> blockZ = state.kernelConfig.getBlockIdx().z;
  tempRead = ReadExpr::extractReadExpr(blockZ);
  const Array *newBZArray = arrayCache.CreateArray(tempRead->updates.root->name+"_2", sizeof(unsigned), true);
  ref<Expr> newblockZ = Expr::createTempRead(newBZArray, sizeof(unsigned) * 8);
  globalReplacements[blockZ] = newblockZ;
  globalThreadCon = AndExpr::create(globalThreadCon, EqExpr::create(blockZ, newblockZ));

  ref<Expr> threadX = state.kernelConfig.getThreadIdx().x;
  tempRead = ReadExpr::extractReadExpr(threadX);
  const Array *newTXArray = arrayCache.CreateArray(tempRead->updates.root->name+"_2", sizeof(unsigned), true);
  ref<Expr> newthreadX = Expr::createTempRead(newTXArray, sizeof(unsigned) * 8);
  globalReplacements[threadX] = newthreadX;
  globalThreadCon = AndExpr::create(globalThreadCon, EqExpr::create(threadX, newthreadX));
  sharedReplacements[threadX] = newthreadX;
  sharedblockCon = EqExpr::create(threadX, newthreadX);

  ref<Expr> threadY = state.kernelConfig.getThreadIdx().y;
  tempRead = ReadExpr::extractReadExpr(threadY);
  const Array *newTYArray = arrayCache.CreateArray(tempRead->updates.root->name+"_2", sizeof(unsigned), true);
  ref<Expr> newthreadY = Expr::createTempRead(newTYArray, sizeof(unsigned) * 8);
  globalReplacements[threadY] = newthreadY;
  globalThreadCon = AndExpr::create(globalThreadCon, EqExpr::create(threadY, newthreadY));
  sharedReplacements[threadY] = newthreadY;
  sharedblockCon = AndExpr::create(sharedblockCon, EqExpr::create(threadY, newthreadY));

  ref<Expr> threadZ = state.kernelConfig.getThreadIdx().z;
  tempRead = ReadExpr::extractReadExpr(threadZ);
  const Array *newTZArray = arrayCache.CreateArray(tempRead->updates.root->name+"_2", sizeof(unsigned), true);
  ref<Expr> newthreadZ = Expr::createTempRead(newTZArray, sizeof(unsigned) * 8);
  globalReplacements[threadZ] = newthreadZ;
  globalThreadCon = AndExpr::create(globalThreadCon, EqExpr::create(threadZ, newthreadZ));
  globalThreadCon = NotExpr::create(globalThreadCon);
  allGlobalNewCon = globalThreadCon;

  sharedReplacements[threadZ] = newthreadZ;
  sharedblockCon = AndExpr::create(sharedblockCon, EqExpr::create(threadZ, newthreadZ));
  sharedblockCon = NotExpr::create(sharedblockCon);
  allSharedNewCon = sharedblockCon;

  bool isLoopIinitialThreadRelated = false;
  bool isLoopIinitialSharedDiff = false;
  if (!loopInfo->initialVal.isNull() && (Expr::isRelatedWithThread(loopInfo->initialVal) || Expr::isRelatedWithBlock(loopInfo->initialVal))) {
    isLoopIinitialSharedDiff = Expr::isRelatedWithThread(loopInfo->initialVal);
    isLoopIinitialThreadRelated = true;
    ref<Expr> oldReadIndex = loopInfo->indexExpr;
    if (isa<CastExpr>(loopInfo->indexExpr)) {
      oldReadIndex = loopInfo->indexExpr->getKid(0);
    }
    int width = oldReadIndex->getWidth();
    const Array *newLIndexArray = arrayCache.CreateArray(loopInfo->indexName+"_2", width / 8, true);
    ref<Expr> newLIndex = Expr::createTempRead(newLIndexArray, width);
    ref<Expr> iterationTimes1 = KLoopInfo::getIterationTimes(loopInfo->indexExpr, loopInfo->increType, loopInfo->increment, loopInfo->initialVal);
    globalReplacements[oldReadIndex] = newLIndex;
    sharedReplacements[oldReadIndex] = newLIndex;

    if (iterationTimes1) {
      ref<Expr> newIncrement = ConstraintManager::replaceReadExpr(loopInfo->increment, globalReplacements);
      ref<Expr> newInitial = ConstraintManager::replaceReadExpr(loopInfo->initialVal, globalReplacements);
      ref<Expr> iterationTimes2 = KLoopInfo::getIterationTimes(newLIndex, loopInfo->increType, newIncrement, newInitial);
      globalBaseConstraints.push_back(EqExpr::create(iterationTimes1, iterationTimes2));
      allGlobalNewCon = AndExpr::create(allGlobalNewCon, EqExpr::create(iterationTimes1, iterationTimes2));

      newIncrement = ConstraintManager::replaceReadExpr(loopInfo->increment, sharedReplacements);
      newInitial = ConstraintManager::replaceReadExpr(loopInfo->initialVal, sharedReplacements);
      iterationTimes2 = KLoopInfo::getIterationTimes(newLIndex, loopInfo->increType, newIncrement, newInitial);
      sharedBaseConstraints.push_back(EqExpr::create(iterationTimes1, iterationTimes2));
      allSharedNewCon = AndExpr::create(allSharedNewCon, EqExpr::create(iterationTimes1, iterationTimes2));
    }
  }

  for (auto con: state.constraints) {
      ref<Expr> replacedCon = ConstraintManager::replaceReadExpr(con, globalReplacements);
      if (replacedCon != con) {
        globalBaseConstraints.push_back(replacedCon);
      }
    replacedCon = ConstraintManager::replaceReadExpr(con, sharedReplacements);
    if (replacedCon != con) {
      sharedBaseConstraints.push_back(replacedCon);
    }
  }

  for(auto pair: state.addressInLoop[loopInfo->indexName]) {
    if (std::get<0>(pair.second) == ExecutionState::MemOp::READ) {
      continue;
    }
    
    ref<Expr> originalAddress = pair.first;
    bool raceMayBeTrue = false;
    Solver::Validity res;
    bool success = true;
    std::string queryStr = "";
    if (std::get<1>(pair.second) == MemoryObject::MemType::GLOBAL) {
      ref<Expr> condition = ConstantExpr::create(1, Expr::Bool);
        ref<Expr> replacedAddress = ConstraintManager::replaceReadExpr(originalAddress, globalReplacements);
        condition = EqExpr::create(replacedAddress, originalAddress);

        std::vector<ref<Expr>> concatExprs;
        std::set<ref<Expr>> visited;
        Expr::findAllConcatExpr(originalAddress, concatExprs, visited);

        if (!concatExprs.empty()) {
          for (auto cexpr: concatExprs) {
            const Array *array = nullptr;
            ref<Expr> indexExpr = ConcatExpr::getArrayIndex(cexpr, &array);
            if (indexExpr) {
              if (state.symbolicArrayMap.find(array->name)!=state.symbolicArrayMap.end()) {
                SymArrayMemoryObject *symmo = state.symbolicArrayMap[array->name];
                if (!symmo->hasDuplicateVal) {
                  ref<Expr> new_cexpr = ConstraintManager::replaceReadExpr(cexpr, globalReplacements);
                  if (cexpr != new_cexpr) {
                    condition = AndExpr::create(condition, NeExpr::create(cexpr, new_cexpr));
                  }
                }
              }
            }
          }
        }
        ref<Expr> raceCondition = AndExpr::create(globalThreadCon, condition);
        success = solver->evaluate(globalBaseConstraints, raceCondition, res,
                                   state.queryMetaData, state.intArrNames,
                                   state.getIOExprs(), true);
        raceMayBeTrue = success && (res == Solver::True || res == Solver::Unknown);
      if (raceMayBeTrue) {
        Query query(globalBaseConstraints, NotExpr::create(raceCondition), state.intArrNames, state.getIOExprs());
        queryStr = solver->getConstraintLog(query);
      }
    } else {
      ref<Expr> condition = ConstantExpr::create(1, Expr::Bool);
        ref<Expr> replacedAddress = ConstraintManager::replaceReadExpr(originalAddress, sharedReplacements);
        condition = EqExpr::create(replacedAddress, originalAddress);

        std::vector<ref<Expr>> concatExprs;
        std::set<ref<Expr>> visited;
        Expr::findAllConcatExpr(originalAddress, concatExprs, visited);

        if (!concatExprs.empty()) {
          for (auto cexpr: concatExprs) {
            const Array *array = nullptr;
            ref<Expr> indexExpr = ConcatExpr::getArrayIndex(cexpr, &array);
            if (indexExpr) {
              if (state.symbolicArrayMap.find(array->name)!=state.symbolicArrayMap.end()) {
                SymArrayMemoryObject *symmo = state.symbolicArrayMap[array->name];
                if (!symmo->hasDuplicateVal) {
                  ref<Expr> new_cexpr = ConstraintManager::replaceReadExpr(cexpr, sharedReplacements);
                  if (cexpr != new_cexpr) {
                    condition = AndExpr::create(condition, NeExpr::create(cexpr, new_cexpr));
                  }
                }
              }
            }
          }
        }
        ref<Expr> raceCondition = AndExpr::create(sharedblockCon, condition);
        success = solver->evaluate(sharedBaseConstraints, raceCondition, res,
                                   state.queryMetaData, state.intArrNames,
                                   state.getIOExprs(), true);
        raceMayBeTrue = success && (res == Solver::True || res == Solver::Unknown);
      if (raceMayBeTrue) {
        Query query(sharedBaseConstraints, NotExpr::create(raceCondition), state.intArrNames, state.getIOExprs());
        queryStr = solver->getConstraintLog(query);
      }
    }
    
    if (raceMayBeTrue) {
      std::string lineId = std::get<2>(pair.second);
      state.dataRaces[lineId].push_back(originalAddress);

      std::string filename = lineId + "_dr";;
      std::string error;
      std::unique_ptr<llvm::raw_fd_ostream> dumpedQueriesFile = klee_open_output_file(outputDir+"/"+filename+".txt", error);
      if (!dumpedQueriesFile) {
        return;
      }
  
      *dumpedQueriesFile << "; start Z3 query\n";
      *dumpedQueriesFile << queryStr;
      *dumpedQueriesFile << "(get-model)\n";
      *dumpedQueriesFile << "(reset)\n";
      *dumpedQueriesFile << "; end Z3 query\n\n";
      dumpedQueriesFile->flush();

      const InstructionInfo &ii = *state.prevPC->info;
      if (!ii.file.empty()) {
        std::string sourceLine = lineId.substr(0, lineId.find('-'));
        sourceLine = sourceLine.substr(0, sourceLine.find('_'));
        klee_message("Bug Detected: %s:%s: data race", ii.file.c_str(), sourceLine.c_str());
      } else {
        std::string assemblyLine = lineId;
        size_t asmPrefix = assemblyLine.find("asm-");
        if (asmPrefix != std::string::npos)
          assemblyLine = assemblyLine.substr(asmPrefix + 4);
        assemblyLine = assemblyLine.substr(0, assemblyLine.find('_'));
        klee_message("Bug Detected: assemblyLine %s: data race", assemblyLine.c_str());
      }
    }
  }
  state.addressInLoop.erase(loopInfo->indexName);
}

void Executor::storeAddressInLoop(ExecutionState &state, KInstruction *ki, ref<Expr> address, const MemoryObject *mo, bool isWrite) {
  llvm::Function *parentFunction = ki->inst->getFunction();
  if (parentFunction) {
    if (parentFunction->getName().str().find("atomic")!=std::string::npos) {
      return;
    }
  }

  if (mo->memType==MemoryObject::MemType::LOCAL || state.addressInLoop.empty()) {
    return;
  }

  bool hasSourceLine = !ki->info->file.empty();
  const StackFrame &lastFrame = state.stack.back();
  unsigned callerSourceLine = 0;
  unsigned callerAsmLine = 0;
  if (auto *callerki = dyn_cast<KInstruction>(&*lastFrame.caller)) {
    callerAsmLine = callerki->info->assemblyLine;
    if (hasSourceLine)
      callerSourceLine = callerki->info->line;
  }

  unsigned grandCallerAsmLine = 0;
  unsigned grandCallerSourceLine = 0;
  if (state.stack.size() >= 2) {
    const StackFrame &grandCallerFrame = state.stack[state.stack.size() - 2];
    if (grandCallerFrame.caller) {
      if (auto *grandCallerKI = dyn_cast<KInstruction>(&*grandCallerFrame.caller)) {
        grandCallerAsmLine = grandCallerKI->info->assemblyLine;
        if (hasSourceLine)
          grandCallerSourceLine = grandCallerKI->info->line;
      }
    }
  }

  std::string lineId = llvm::utostr(ki->info->assemblyLine);
  if (callerAsmLine) {
    lineId+="_"+llvm::utostr(callerAsmLine);
  }
  if (grandCallerAsmLine) {
    lineId+="_"+llvm::utostr(grandCallerAsmLine);
  }
  lineId = "asm-"+lineId;

  std::string tmp = "";
  if (hasSourceLine && ki->info->line > 0) {
    tmp = llvm::utostr(ki->info->line) ;
  }
  if (callerSourceLine > 0) {
    tmp+="_"+llvm::utostr(callerSourceLine);
  }
  if (grandCallerSourceLine > 0) {
    tmp+="_"+llvm::utostr(grandCallerSourceLine);
  }
  if (!tmp.empty()) {
    lineId = tmp+"-"+lineId;
  }

  for (auto &pair : state.addressInLoop) {
    auto &addressMap = pair.second;
    if (addressMap.find(address) == addressMap.end())
      addressMap[address] = std::make_tuple(isWrite ? ExecutionState::MemOp::WRITE : ExecutionState::MemOp::READ, mo->memType, lineId);
    else if (std::get<0>(addressMap[address])==ExecutionState::MemOp::READ && isWrite)
      addressMap[address] = std::make_tuple(ExecutionState::MemOp::BOTH, mo->memType, lineId);
    else if (std::get<0>(addressMap[address])==ExecutionState::MemOp::WRITE && !isWrite)
      std::get<0>(addressMap[address]) = ExecutionState::MemOp::BOTH;
  }
}

bool isInSymArray(KInstruction *target, ref<Expr> address) {
  if (!llvm::isa<klee::ConstantExpr>(address)) {
    return true;
  }

  llvm::Instruction *inst = target->inst;
  if (!inst)
    return false;
  
  // Get the previous instruction in the same BasicBlock
  llvm::BasicBlock::iterator it(inst);
  if (it == inst->getParent()->begin())
    return false; // no previous instruction

  --it;
  llvm::Instruction *prevInst = &*it;

  if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(prevInst)) {
    auto srcTy = gep->getSourceElementType();

    if (srcTy->isIntegerTy() || srcTy->isFloatingPointTy()) {
      return true;
    } else if (srcTy->isStructTy()) {
      return false;
    }
  }
  return false;
}

void Executor::modifyArrayBound(ExecutionState &state, ref<Expr> value, SymArrayMemoryObject *symmo, KInstruction *i) {
  if (!symmo) return;

  if (auto valueCE = dyn_cast<ConstantExpr>(value)) {
    if (value->getWidth() > 64) {
      return;
    }
    uint64_t rawValue = valueCE->getZExtValue();
    if (rawValue != 0) {
      if (symmo->isFloat) {
        if (value->getWidth() == 32) {
          float x = ConstantExpr::getFloatFromConstantExpr(value);
          if (x <= INT32_MIN) {
            return;
          }
        }
        if (value->getWidth() == 64) {
          double x = ConstantExpr::getDoubleFromConstantExpr(value);
          if (x <= INT64_MIN) {
            return;
          }
        }
      }
    }
    symmo->minVal = value;
    return;
  } else if (isa<ReadExpr>(value) || isa<ConcatExpr>(value)) {
    const Array *array = nullptr;
    ref<Expr> index = ConcatExpr::getArrayIndex(value, &array);
    if (!index.isNull()) {
      if (array->isIntVar) {
        if (symmo->minVal.isNull())
          symmo->minVal = value;
        symmo->maxVal = value;
      } else if (state.symbolicArrayMap.find(array->name) != state.symbolicArrayMap.end()) {
        SymArrayMemoryObject *sourceSymmo = state.symbolicArrayMap[array->name];
        if (symmo->minVal.isNull())
          symmo->minVal = sourceSymmo->minVal;
        else {
          if (isa<ConstantExpr>(symmo->minVal) && (!sourceSymmo->minVal.isNull() && isa<ConstantExpr>(sourceSymmo->minVal))) {
            auto minValCE0 = dyn_cast<ConstantExpr>(symmo->minVal);
            int minV0 = minValCE0->getZExtValue();
            auto minValCE1 = dyn_cast<ConstantExpr>(sourceSymmo->minVal);
            int minV1 = minValCE1->getZExtValue();
            if (minV1 < minV0) {
              symmo->minVal = sourceSymmo->minVal;
            }
          }
        }
        if (symmo->maxVal.isNull())
          symmo->maxVal = sourceSymmo->maxVal;
        else {
          if (isa<ConstantExpr>(symmo->maxVal) && (!sourceSymmo->maxVal.isNull() && isa<ConstantExpr>(sourceSymmo->maxVal))) {
            auto maxValCE0 = dyn_cast<ConstantExpr>(symmo->maxVal);
            int maxV0 = maxValCE0->getZExtValue();
            auto maxValCE1 = dyn_cast<ConstantExpr>(sourceSymmo->maxVal);
            int maxV1 = maxValCE1->getZExtValue();
            if (maxV1 > maxV0) {
              symmo->maxVal = sourceSymmo->maxVal;
            }
          }
        }
      }
    } else {
      return;
    }
  }

  if (!isa<AddExpr>(value)) {
    return;
  }

  bool hasConstantMax = false;
  if (!symmo->maxVal.isNull()) {
    if (auto maxCE = dyn_cast<ConstantExpr>(symmo->maxVal)) {
      int64_t maxVal = maxCE->getZExtValue();
      if (maxVal < INT32_MAX) {
        hasConstantMax = true;
        if (auto minCE = dyn_cast<ConstantExpr>(symmo->minVal)) {
          int64_t minVal = minCE->getZExtValue();
          if (minVal == maxVal) {
            hasConstantMax = false;
          }
        }
      }
    }
  }
  if (hasConstantMax) return;

  ref<Expr> addOp = nullptr;
  ref<Expr> arrayOp = nullptr;
  const Array *array0 = nullptr;
  ref<Expr> index0 = ConcatExpr::getArrayIndex(value->getKid(0), &array0);
  bool addSeond = false;
  const Array *array1 = nullptr;
  ref<Expr> index1 = ConcatExpr::getArrayIndex(value->getKid(1), &array1);
  if (!index0.isNull() && array0->name == symmo->arrayName->name) {
    arrayOp = value->getKid(0);
    addOp = value->getKid(1);
    addSeond = true;
  } else if (!index1.isNull() && array1->name == symmo->arrayName->name) {
    arrayOp = value->getKid(1);
    addOp = value->getKid(0);
  }

  if (addOp.isNull() || arrayOp.isNull()) return;
  
  bool isInloop = false;
  KLoopInfo *info;
  if (i) {
    llvm::BasicBlock *BB = i->inst->getParent();
    if (BB) {
      for (auto const &entry : state.loopInfoMap) {
        info = entry.second;
        if (info && info->bodyBlocks.count(BB)) {
          isInloop = true;
          break;
        }
      }
    }
  }

  ref<Expr> arrBIndex = addSeond ? index1 : index0;
  const Array *arrBName = addSeond ? array1 : array0;
  ref<Expr> arrBMax = nullptr;
  if (!arrBIndex.isNull()) {
    if (state.symbolicArrayMap.find(arrBName->name) != state.symbolicArrayMap.end()) {
      SymArrayMemoryObject *arrBSymmo = state.symbolicArrayMap[arrBName->name];
      if (!arrBSymmo->maxVal.isNull()) {
        arrBMax = arrBSymmo->maxVal;
      }
    }
  }
  if (isInloop) {
    ref<Expr> loopCount = nullptr;
    if (info->increType == KLoopInfo::IncreType::ADD && !info->exitExpr.isNull()) {
      loopCount = info->exitExpr;
    } else if (info->increType == KLoopInfo::IncreType::SUB && !info->initialVal.isNull()) {
      loopCount = info->initialVal;
    }
    if (!loopCount.isNull()) {
      if (!addOp.isNull() && isa<ConstantExpr>(addOp)) {
       ref<Expr> toAdd = MulExpr::create(addOp, SExtExpr::create(loopCount, addOp->getWidth()));
        if (!symmo->maxVal.isNull()) {
          symmo->maxVal = AddExpr::create(SExtExpr::create(symmo->maxVal, toAdd->getWidth()), toAdd);
        } else {
          symmo->maxVal = toAdd;
        }
        return;
      }
    }
  } 
    if (!arrBIndex.isNull()) {
      if (!arrBMax.isNull()) {
        if (!symmo->maxVal.isNull()) {
          symmo->maxVal = AddExpr::create(symmo->maxVal, SExtExpr::create(arrBMax, symmo->maxVal->getWidth()));
        } else {
          symmo->maxVal = arrBMax;
        }
      }
    } else {
      if (!symmo->maxVal.isNull()) {
        symmo->maxVal = AddExpr::create(SExtExpr::create(symmo->maxVal, addOp->getWidth()), addOp);
      } else {
        symmo->maxVal = addOp;
      }
    }
}

void Executor::executeMemoryOperation(ExecutionState &state,
                                      bool isWrite,
                                      ref<Expr> address,
                                      ref<Expr> value /* undef if read */,
                                      KInstruction *target, unsigned loadBytes, ref<Expr> selectAddress) {                                   

  Expr::Width type;
  unsigned bytes;
  if (loadBytes) {
    bytes = loadBytes;
    type = bytes * 8;
  } else {
    type = (isWrite ? value->getWidth() : 
                     getWidthForLLVMType(target->inst->getType()));
    bytes = Expr::getMinBytesForWidth(type);
  }

  if (SimplifySymIndices) {
    if (!isa<ConstantExpr>(address))
      address = ConstraintManager::simplifyExpr(state.constraints, address);
    if (isWrite && !isa<ConstantExpr>(value))
      value = ConstraintManager::simplifyExpr(state.constraints, value);
  }

  address = optimizer.optimizeExpr(address, true);

  ObjectPair op;
  bool success;
  solver->setTimeout(coreSolverTimeout);

  bool resolveSingleObject = SingleObjectResolution;

  if (resolveSingleObject && !isa<ConstantExpr>(address)) {
    // Address is symbolic

    resolveSingleObject = false;
    auto base_it = state.base_addrs.find(address);
    if (base_it == state.base_addrs.end()) {
      if (auto extractExpr = dyn_cast<ExtractExpr>(address)) {
        if (address->getWidth() == 32) {
          base_it = state.base_addrs.find(extractExpr->expr);
        }
      }
    }
    if (base_it == state.base_addrs.end() && selectAddress) {
      base_it = state.base_addrs.find(selectAddress);
    }
    if (base_it != state.base_addrs.end()) {
      // Concrete address found in the map, now find the associated memory
      // object
      if (!state.addressSpace.resolveOne(state, solver.get(), base_it->second, op,
                                         success) ||
          !success) {
        klee_warning("Failed to resolve concrete address from the base_addrs "
                     "map to a memory object");
      } else {
        // We have resolved the stored concrete address to a memory object.
        // Now let's see if we can prove an overflow - we are only interested in
        // two cases: either we overflow and it's a bug or we don't and we carry
        // objects
        resolveSingleObject = true;
      }
    }
  } else {
    resolveSingleObject = false;
  }

  if (!resolveSingleObject) {
    if (!state.addressSpace.resolveOne(state, solver.get(), address, op, success) || !success) {
      if (isa<ConstantExpr>(address)) {
        uint64_t addressValue = dyn_cast<ConstantExpr>(address)->getZExtValue();
        if (addressValue == 0) {
          terminateStateOnProgramError(state, "null pointer dereference", StateTerminationType::Ptr);
          return;
        }
      }
      terminateStateOnProgramError(state, "read/write: not find object", StateTerminationType::ReportError);
      return;
    }
    solver->setTimeout(time::Span());

    if (success) {
      const MemoryObject *mo = op.first;

      if (MaxSymArraySize && mo->size >= MaxSymArraySize) {
        address = toConstant(state, address, "max-sym-array-size");
      }

      ref<Expr> offset = mo->getOffsetExpr(address);
      ref<Expr> check = mo->getBoundsCheckOffset(offset, bytes);
      check = AndExpr::create(check, SgeExpr::create(offset, ConstantExpr::create(0, offset->getWidth())));

      if (state.tensorSizesMap.find(mo->name)!=state.tensorSizesMap.end()) {
        if (auto offsetCE = dyn_cast<ConstantExpr>(offset)) {
          uint64_t offsetVal = offsetCE->getZExtValue();
          uint64_t index  = offsetVal/8;
          uint64_t startBit = (offsetVal%8)*8;
          TensorSizesMemoryObject *tensorSizesMO = state.tensorSizesMap[mo->name];
          if (state.symbolicArrayMap.find(tensorSizesMO->tensorName)==state.symbolicArrayMap.end()) {
            terminateStateOnProgramError(state, "did not find tensor object", StateTerminationType::ReportError);
            return;
          }

          SymArrayMemoryObject *symmo = state.symbolicArrayMap[tensorSizesMO->tensorName];
          ref<Expr> result;
          if (tensorSizesMO->name->name.find(".vec")!=std::string::npos) {
            if (isWrite) {
              if (startBit != 0) {
                terminateStateOnProgramError(state, "not support to write part of element in symbolic vector", StateTerminationType::ReportError);
                return;
              }
              tensorSizesMO->elements[index] = value;
              return;
            } else {
              if (tensorSizesMO->elements.find(index) != tensorSizesMO->elements.end()) {
                result = ExtractExpr::create(tensorSizesMO->elements[index], startBit, bytes*8);
                bindLocal(target, state, result);
                return;
              }
            }
          }
          if (tensorSizesMO->name->name.find("stride")!=std::string::npos) {
            if (isWrite) {
              if (startBit != 0) {
                terminateStateOnProgramError(state, "not support to write part of element in symbolic array", StateTerminationType::ReportError);
                return;
              }
              symmo->strides[index] = value;
              tensorSizesMO->elements[index] = value;
              return;
            } else {
              if (symmo->strides.find(index) == symmo->strides.end()) {
                std::string strideName = symmo->arrayName->name + ".stride[" + std::to_string(index)+"]";
                auto stridepair = createSizeSymbol(state, strideName, true);
                ref<Expr> strideExpr = stridepair.second; 
                symmo->strides[index] = strideExpr;
              }
              result = ExtractExpr::create(symmo->strides[index], startBit, bytes*8);
              bindLocal(target, state, result);
              return;
            }
          } else {
            if (isWrite) {
              if (startBit != 0) {
                terminateStateOnProgramError(state, "not support to write part of element in symbolic array", StateTerminationType::ReportError);
                return;
              }
              symmo->dimensionSize[index] = value;
              tensorSizesMO->elements[index] = value;
              return;
            } else {
              if (symmo->dimensionSize.find(index) == symmo->dimensionSize.end()) {
                std::string sizeName = symmo->arrayName->name + ".size[" + std::to_string(index)+"]";
                auto sizepair = createSizeSymbol(state, sizeName);
                ref<Expr> sizeExpr = sizepair.second; 
                addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
                symmo->dimensionSize[index] = sizeExpr;
              }
              result = ExtractExpr::create(symmo->dimensionSize[index], startBit, bytes*8);
              bindLocal(target, state, result);
              return;
            }
          }
        } else {
          terminateStateOnProgramError(state, "tensor sizes offset is not constant", StateTerminationType::ReportError);
          return;
        }
      }

      std::string symName = mo->name;
      if ((mo->name.empty() || mo->name=="unnamed") && state.symAddressMap.find(mo->address) != state.symAddressMap.end()) {
        ref<Expr> newAddress = ConstantExpr::createPointer(state.symAddressMap[mo->address]);
        // we create tensor as a object with large size, but tensor struct is a ptr
        if (isInSymArray(target, address)) {
          symName = state.symNames[state.symAddressMap[mo->address]];
          offset = SubExpr::create(address, newAddress);
        }
      }
      if (state.symbolicArrayMap.find(symName)!=state.symbolicArrayMap.end()) {
        storeAddressInLoop(state, target, address, mo, isWrite);
        SymArrayMemoryObject *symmo = state.symbolicArrayMap[symName];
        bool isLoadOrStore = false;
        if (target) {
          isLoadOrStore = llvm::isa<llvm::LoadInst>(target->inst) || llvm::isa<llvm::StoreInst>(target->inst);
        }
        if (isLoadOrStore && bytes >= 1 && bytes <= 8) {
          llvm::Function *parentFunction = target->inst->getFunction();
          if (!parentFunction || !isMemFuncName(parentFunction->getName().str())) {
            state.intArrNames[symName] = bytes;
          }
        }

        ref<Expr> sizeConstraint;
        setElementSize(state, symmo);
        if (symmo->elementSize) {
          ref<Expr> sizeBytes = MulExpr::create(symmo->size, ZExtExpr::create(symmo->elementSize, symmo->size->getWidth()));
          sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(offset, sizeBytes, bytes);
        } else {
          sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(offset, symmo->size, bytes);
        }
        sizeConstraint = AndExpr::create(sizeConstraint, SgeExpr::create(offset, ConstantExpr::create(0, offset->getWidth())));

        StatePair branches = fork(state, sizeConstraint, true, BranchType::MemOp);
        ExecutionState *bound = branches.first;
        const ObjectState *os = op.second;

        if (bound) {
          if (isWrite) {
            if (os->readOnly) {
              terminateStateOnProgramError(*bound, "memory error: object read only",
                                          StateTerminationType::ReadOnly);
            } else {
              ObjectState *wos = bound->addressSpace.getWriteable(mo, os);
              wos->write(offset, value);
                modifyArrayBound(state, value, symmo, target);
            }
          } else {
            ref<Expr> result = os->read(offset, type);
            if (!isa<ConstantExpr>(result)) {
              if (!symmo->maxVal.isNull()) {
                if (symmo->elementSize.isNull() || (isa<ConstantExpr>(symmo->elementSize) && dyn_cast<ConstantExpr>(symmo->elementSize)->getZExtValue()>=bytes)) {
                  addConstraint(state, SleExpr::create(result, SExtExpr::create(symmo->maxVal, result->getWidth())));
                }
              }
              if (!symmo->minVal.isNull()) {
                if (symmo->elementSize.isNull() || (isa<ConstantExpr>(symmo->elementSize) && dyn_cast<ConstantExpr>(symmo->elementSize)->getZExtValue()>=bytes)) {
                  addConstraint(state, SgeExpr::create(result, SExtExpr::create(symmo->minVal, result->getWidth())));
                }
              }
            } 
            bindLocal(target, *bound, result);
          }
        }
        
        ExecutionState *unbound = branches.second;
        if (unbound) {
          // llvm::outs() << address << " " << mo->name << " " << sizeConstraint << "\n";
          terminateStateOnProgramError(*unbound, "out of bound pointer", StateTerminationType::Ptr);
        }
        return;
      }
      check = optimizer.optimizeExpr(check, true);

      bool inBounds;
      solver->setTimeout(coreSolverTimeout);
      bool success = solver->mustBeTrue(state.constraints, check, inBounds,
                                        state.queryMetaData, state.intArrNames);
      solver->setTimeout(time::Span());
      if (!success) {
        state.pc = state.prevPC;
        terminateStateOnSolverError(state, "Query timed out (bounds check).");
        return;
      }

      if (inBounds) {
        const ObjectState *os = op.second;
        if (isWrite) {
          if (os->readOnly) {
            terminateStateOnProgramError(state, "memory error: object read only",
                                         StateTerminationType::ReadOnly);
          } else {
            ObjectState *wos = state.addressSpace.getWriteable(mo, os); 
            wos->write(offset, value);
          }
        } else {
          ref<Expr> result = os->read(offset, type);
          if (interpreterOpts.MakeConcreteSymbolic)
            result = replaceReadWithSymbolic(state, result);

          llvm::Function *parentFunction = target->inst->getFunction();
          bool isLoadIr = false;
          if (target) {
            isLoadIr = llvm::isa<llvm::LoadInst>(target->inst);
          }
          if (isLoadIr && bytes > 1 && bytes <=8 && (!parentFunction || !isMemFuncName(parentFunction->getName().str()))) {
            ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(result);
            if (readExpr && (readExpr->updates.root->name.find("const_arr")!=std::string::npos || readExpr->updates.root->name.find("sym_arr")!=std::string::npos)) {
              llvm::Type *type = target->inst->getType();
              if (type->isIntegerTy()) {
                std::string aName = readExpr->updates.root->name;
                state.intArrNames[aName] = bytes;
              }
            }
          }
          bindLocal(target, state, result);
        }

        return;
      } 
      else if (!isa<ConstantExpr>(address)) {
        // llvm::outs() << mo->name << " address " << address << " mo->address: " << mo->address << " offset " << offset << " bytes " << bytes << " size " << mo->size << "\n";
        terminateStateOnProgramError(state, "cannot handle memory operation", StateTerminationType::ReportError);
        return;
      }
    }
  }


  address = optimizer.optimizeExpr(address, true);
  ResolutionList rl;
  bool incomplete = false;

  if (!resolveSingleObject) {
    solver->setTimeout(coreSolverTimeout);
    incomplete = state.addressSpace.resolve(state, solver.get(), address, rl, 0,
                                            coreSolverTimeout);
    solver->setTimeout(time::Span());
  } else {
    rl.push_back(op); // we already have the object pair, no need to look for it
  }
  // XXX there is some query wasteage here. who cares?
  ExecutionState *unbound = &state;
  SymArrayMemoryObject *symmo = nullptr;
  
  for (ResolutionList::iterator i = rl.begin(), ie = rl.end(); i != ie; ++i) {
    const MemoryObject *mo = i->first;
    const ObjectState *os = i->second;
    ref<Expr> inBounds = mo->getBoundsCheckPointer(address, bytes);
    ref<Expr> offset = mo->getOffsetExpr(address);
    inBounds = AndExpr::create(inBounds, SgeExpr::create(offset, ConstantExpr::create(0, offset->getWidth())));

    if (state.tensorSizesMap.find(mo->name)!=state.tensorSizesMap.end()) {
      if (auto offsetCE = dyn_cast<ConstantExpr>(offset)) {
        uint64_t offsetVal = offsetCE->getZExtValue();
        uint64_t index  = offsetVal/8;
        uint64_t startBit = (offsetVal%8)*8;
        TensorSizesMemoryObject *tensorSizesMO = state.tensorSizesMap[mo->name];
        if (state.symbolicArrayMap.find(tensorSizesMO->tensorName)==state.symbolicArrayMap.end()) {
          terminateStateOnProgramError(state, "did not find tensor object", StateTerminationType::ReportError);
          return;
        }

        symmo = state.symbolicArrayMap[tensorSizesMO->tensorName];
        ref<Expr> result;
        if (tensorSizesMO->name->name.find(".vec")!=std::string::npos) {
          if (isWrite) {
            if (startBit != 0) {
              terminateStateOnProgramError(state, "not support to write part of element in symbolic vector", StateTerminationType::ReportError);
              return;
            }
            tensorSizesMO->elements[index] = value;
            return;
          } else {
            if (tensorSizesMO->elements.find(index) != tensorSizesMO->elements.end()) {
              result = ExtractExpr::create(tensorSizesMO->elements[index], startBit, bytes*8);
              bindLocal(target, state, result);
              return;
            }
          }
        }
        if (tensorSizesMO->name->name.find("stride")!=std::string::npos) {
          if (isWrite) {
            if (startBit != 0) {
              terminateStateOnProgramError(state, "not support to write part of element in symbolic array", StateTerminationType::ReportError);
              return;
            }
            symmo->strides[index] = value;
            tensorSizesMO->elements[index] = value;
            return;
          } else {
            if (symmo->strides.find(index) == symmo->strides.end()) {
              std::string strideName = symmo->arrayName->name + ".stride[" + std::to_string(index)+"]";
              auto stridepair = createSizeSymbol(state, strideName, true);
              ref<Expr> strideExpr = stridepair.second; 
              symmo->strides[index] = strideExpr;
            }
            result = ExtractExpr::create(symmo->strides[index], startBit, bytes*8);
            bindLocal(target, state, result);
            return;
          }
        } else {
          if (isWrite) {
            if (startBit != 0) {
              terminateStateOnProgramError(state, "not support to write part of element in symbolic array", StateTerminationType::ReportError);
              return;
            }
            symmo->dimensionSize[index] = value;
            tensorSizesMO->elements[index] = value;
            return;
          } else {
            if (symmo->dimensionSize.find(index) == symmo->dimensionSize.end()) {
              std::string sizeName = symmo->arrayName->name + ".size[" + std::to_string(index)+"]";
              auto sizepair = createSizeSymbol(state, sizeName);
              ref<Expr> sizeExpr = sizepair.second; 
              addConstraint(state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
              symmo->dimensionSize[index] = sizeExpr;
            }
            result = ExtractExpr::create(symmo->dimensionSize[index], startBit, bytes*8);
            bindLocal(target, state, result);
            return;
          }
        }
      } else {
        terminateStateOnProgramError(state, "tensor sizes offset is not constant", StateTerminationType::ReportError);
        return;
      }
    }

    std::string symName = mo->name;
    if ((mo->name.empty() || mo->name=="unnamed") && state.symAddressMap.find(mo->address) != state.symAddressMap.end()) {
      ref<Expr> newAddress = ConstantExpr::createPointer(state.symAddressMap[mo->address]);
      if (isInSymArray(target, address)) {
        symName = state.symNames[state.symAddressMap[mo->address]];
        offset = SubExpr::create(address, newAddress);
      }
    }
    if (state.symbolicArrayMap.find(symName)!=state.symbolicArrayMap.end()) {
      storeAddressInLoop(state, target, address, mo, isWrite);
      symmo = state.symbolicArrayMap[symName];
      bool isLoadOrStore = false;
      if (target) {
        isLoadOrStore = llvm::isa<llvm::LoadInst>(target->inst) || llvm::isa<llvm::StoreInst>(target->inst);
      }
      if (isLoadOrStore && target && bytes >= 1 && bytes <= 8) {
        llvm::Function *parentFunction = target->inst->getFunction();
        if (!parentFunction || !isMemFuncName(parentFunction->getName().str())) {
          state.intArrNames[symName] = bytes;
        }
      }
      ref<Expr> sizeConstraint;
      setElementSize(state, symmo);
      if (!symmo->elementSize.isNull() && !(sharedAddresses.find(mo->getBaseExpr()) != sharedAddresses.end() && sharedAddresses[mo->getBaseExpr()].second)) {
        ref<Expr> sizeBytes = MulExpr::create(symmo->size, ZExtExpr::create(symmo->elementSize, symmo->size->getWidth()));
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(offset, sizeBytes, bytes);
      } else {
        sizeConstraint = MemoryObject::getBoundsCheckOffsetWithSize(offset, symmo->size, bytes);
      }
      sizeConstraint = AndExpr::create(sizeConstraint, SgeExpr::create(offset, ConstantExpr::create(0, offset->getWidth())));
      inBounds = sizeConstraint;
    }
    
    StatePair branches = fork(*unbound, inBounds, true, BranchType::MemOp);
    ExecutionState *bound = branches.first;

    // bound can be 0 on failure or overlapped 
    if (bound) {
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnProgramError(*bound, "memory error: object read only",
                                       StateTerminationType::ReadOnly);
        } else {
          ObjectState *wos = bound->addressSpace.getWriteable(mo, os);
          wos->write(mo->getOffsetExpr(address), value);
          if (symmo) {
              modifyArrayBound(state, value, symmo, target);
          }
        }
      } else {
        ref<Expr> result = os->read(mo->getOffsetExpr(address), type);
        llvm::Function *parentFunction = target->inst->getFunction();
        bool isLoadIr = false;
        if (target) {
          isLoadIr = llvm::isa<llvm::LoadInst>(target->inst);
        }
        if (isLoadIr && bytes > 1 && bytes <= 8 && (!parentFunction || !isMemFuncName(parentFunction->getName().str()))) {
          ref<ReadExpr> readExpr = ReadExpr::extractReadExpr(result);
          if (readExpr && (readExpr->updates.root->name.find("const_arr")!=std::string::npos || readExpr->updates.root->name.find("sym_arr")!=std::string::npos)) {
            llvm::Type *type = target->inst->getType();
            if (type->isIntegerTy()) {
              std::string aName = readExpr->updates.root->name;
              state.intArrNames[aName] = bytes;
            }
          }
        }

        if (symmo) {
          if (!isa<ConstantExpr>(result)) {
            if (!symmo->maxVal.isNull()) {
              if (symmo->elementSize.isNull() || (isa<ConstantExpr>(symmo->elementSize) && dyn_cast<ConstantExpr>(symmo->elementSize)->getZExtValue()>=bytes)) {
                addConstraint(*bound, SleExpr::create(result, SExtExpr::create(symmo->maxVal, result->getWidth())));
              }
            }
            if (!symmo->minVal.isNull()) {
              if (symmo->elementSize.isNull() || (isa<ConstantExpr>(symmo->elementSize) && dyn_cast<ConstantExpr>(symmo->elementSize)->getZExtValue()>=bytes)) {
                addConstraint(*bound, SgeExpr::create(result, SExtExpr::create(symmo->minVal, result->getWidth())));
              }
            }
          } 
        }
        
        bindLocal(target, *bound, result);
      }
    }

    unbound = branches.second;
    if (!unbound)
      break;
  }

  // XXX should we distinguish out of bounds and overlapped cases?
  if (unbound) {
    if (incomplete) {
      terminateStateOnSolverError(*unbound, "Query timed out (resolve).");
    } else {
      if (auto CE = dyn_cast<ConstantExpr>(address)) {
        std::uintptr_t ptrval = CE->getZExtValue();
        auto ptr = reinterpret_cast<void *>(ptrval);
        if (ptrval < MemoryManager::pageSize) {
          terminateStateOnProgramError(
              *unbound, "memory error: null page access",
              StateTerminationType::Ptr, getAddressInfo(*unbound, address));
          return;
        } else if (MemoryManager::isDeterministic) {
          using kdalloc::LocationInfo;
          auto li = unbound->heapAllocator.locationInfo(ptr, bytes);
          if (li == LocationInfo::LI_AllocatedOrQuarantined) {
            auto base = reinterpret_cast<std::uintptr_t>(li.getBaseAddress());
            auto baseExpr = Expr::createPointer(base);
            ObjectPair op;
            if (!unbound->addressSpace.resolveOne(baseExpr, op)) {
              terminateStateOnProgramError(
                  *unbound, "memory error: use after free",
                  StateTerminationType::Ptr, getAddressInfo(*unbound, address));
              return;
            }
          }
        }
      }
      
      terminateStateOnProgramError(
          *unbound, "out of bound pointer",
          StateTerminationType::Ptr);
    }
  }
}

void Executor::executeMakeSymbolic(ExecutionState &state, 
                                   const MemoryObject *mo,
                                   const std::string &name) {
  if (!replayKTest) {
    // Find a unique name for this array.  First try the original name,
    // or if that fails try adding a unique identifier.
    unsigned id = 0;
    std::string uniqueName = name;
    while (!state.arrayNames.insert(uniqueName).second) {
      uniqueName = name + "_" + llvm::utostr(++id);
    }
    const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
    bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);
    
    auto found = seedMap.find(&state);
    if (found != seedMap.end()) {
      // In seed mode we need to add this as binding

      for (SeedInfo &si : found->second) {
        KTestObject *obj = si.getNextInput(mo, NamedSeedMatching);

        if (!obj) {
          if (AllowSeedExtension) {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values = std::vector<unsigned char>(mo->size, '\0');
          } else /*if (!AllowSeedExtension)*/ {
            terminateStateOnUserError(state,
                                      "ran out of inputs during seeding");
            break;
          }
        } else {
          /* The condition below implies obj->numBytes != mo->size */
          if ((obj->numBytes < mo->size && !AllowSeedExtension) ||
              (obj->numBytes > mo->size && !AllowSeedTruncation)) {
            std::stringstream msg;
	    msg << "replace size mismatch: "
		<< mo->name << "[" << mo->size << "]"
		<< " vs " << obj->name << "[" << obj->numBytes << "]"
		<< " in test\n";
            terminateStateOnUserError(state, msg.str());
            break;
          } else {
            /* Either sizes are equal or seed extension/trucation is allowed */
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values.insert(values.begin(), obj->bytes,
                          obj->bytes + std::min(obj->numBytes, mo->size));
              for (unsigned i = obj->numBytes; i < mo->size; ++i)
                values.push_back('\0');
          }
        }
      }
    }
  } else {
    ObjectState *os = bindObjectInState(state, mo, false);
    if (replayPosition >= replayKTest->numObjects) {
      terminateStateOnUserError(state, "replay count mismatch");
    } else {
      KTestObject *obj = &replayKTest->objects[replayPosition++];
      if (obj->numBytes != mo->size) {
        terminateStateOnUserError(state, "replay size mismatch");
      } else {
        for (unsigned i=0; i<mo->size; i++)
          os->write8(i, obj->bytes[i]);
      }
    }
  }
}

MemoryObject* Executor::loadTensorConfig(ExecutionState *state, ParameterValue pval, std::string symName, unsigned argSize) {
  ref<Expr> totalSizeExpr = ConstantExpr::create(1, Expr::Int64, false);
  std::map<int, ref<Expr>> dimensionSize;
  for (unsigned i = 0; i < pval.dim; i++) {
    ref<Expr> dimSizeExpr;
    if (!pval.shape.empty() && pval.shape.find(i)!=pval.shape.end() && pval.shape[i] != -1) {
      dimSizeExpr = ConstantExpr::create(pval.shape[i], Expr::Int64, false);
    } else {
      std::string sizeName = symName + ".size[" + std::to_string(i)+"]";
      auto sizepair = createSizeSymbol(*state, sizeName);
      dimSizeExpr = sizepair.second;
      addConstraint(*state, UgeExpr::create(dimSizeExpr, ConstantExpr::create(1, dimSizeExpr->getWidth())));
    }
    dimensionSize[i] = dimSizeExpr;
    totalSizeExpr = MulExpr::create(totalSizeExpr, dimSizeExpr);
  }
  if (!isa<ConstantExpr>(totalSizeExpr)) {
    ref<Expr> maxValueExpr = ConstantExpr::create((1ULL << 63) - 1, 64);
    addConstraint(*state, UleExpr::create(totalSizeExpr, maxValueExpr));
  }

  ref<Expr> strideExpr =  ConstantExpr::create(1, Expr::Int64, false);
  std::map<int, ref<Expr>> strides;
  for (int i = pval.dim - 1; i >= 0; i--) {
    strides[i] = strideExpr;
    strideExpr = MulExpr::create(strideExpr, dimensionSize[i]);
  }
  
  MemoryObject *mo =
    memory->allocate(argSize, /*isLocal=*/false, /*isGlobal=*/false,
                    state, /*allocSite=*/state->pc->inst,
                    /*alignment=*/8);    
  mo->setName(symName);
  const Array *array = arrayCache.CreateArray(symName, mo->size, false);
  bindObjectInState(*state, mo, false, array);
  state->addSymbolic(mo, array);

  SymArrayMemoryObject *symmo = new SymArrayMemoryObject(mo->address, array, nullptr, totalSizeExpr);
  symmo->shapeSize = ConstantExpr::create(pval.dim, Expr::Int64, false);
  symmo->dimensionSize = dimensionSize;
  symmo->strides = strides;
  state->symbolicArrayMap[symName] = symmo;

  if (pval.elementType.empty()){
    std::string elementSizeName = symName + ".item_size";
    auto pair2 = createSizeSymbol(*state, elementSizeName);
    ref<Expr> elementSizeExpr = pair2.second;
    symmo->elementSize = elementSizeExpr;
    addConstraint(*state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
    addConstraint(*state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));
  } else {
    ref<Expr> scalarType = getScalarTypeByStr(pval.elementType);
    symmo->scalarType = scalarType;
    unsigned elementSize = getTensorTypeBytes(scalarType);
    symmo->elementSize = ConstantExpr::create(elementSize, Expr::Int32, false);
    if (pval.elementType.find("bool")!=std::string::npos) {
      symmo->maxVal = ConstantExpr::create(1, Expr::Int32);
      symmo->minVal = ConstantExpr::create(0, Expr::Int32);
    } else if (elementSize < 4 && elementSize > 0) {
      int ewidth = elementSize*8;
      if (pval.elementType.find("uint")!=std::string::npos) {
        symmo->maxVal = ConstantExpr::create((1ULL << ewidth) - 1, Expr::Int32);
        symmo->minVal = ConstantExpr::create(0, Expr::Int32);
      } else {
        symmo->maxVal = ConstantExpr::create((1ULL << (ewidth-1)) - 1, Expr::Int32);
        symmo->minVal = ConstantExpr::alloc(1ULL << (ewidth-1), ewidth, true);
      }
    }
    if (pval.elementType.find("int")!=std::string::npos) {
      symmo->isFloat = false;
    }
    if (pval.elementType.find("float")!=std::string::npos) {
      symmo->isFloat = true;
    }
  }

  if (pval.maxVal < INT32_MAX) {
    symmo->maxVal = ConstantExpr::create(pval.maxVal, Expr::Int32, true);
  }
  if (pval.minVal > INT32_MIN) {
    uint64_t uval = static_cast<uint32_t>(pval.minVal); 
    symmo->minVal = ConstantExpr::create(uval, Expr::Int32, true);
  }
  symmo->hasDuplicateVal = pval.hasDuplicateVal;
  return mo;
}

double Executor::parseCoefficient(const std::string& token) {
    size_t pos = token.find('*');
    if (pos == std::string::npos)
        return 1.0; // implicit coefficient

    return std::stod(token.substr(0, pos));
}

double parseDividend(const std::string& token) {
    size_t pos = token.find('/');
    if (pos == std::string::npos)
        return 1.0;

    return std::stod(token.substr(pos+1, token.length()));
}

bool Executor::isNumber(const std::string& s) {
    static const std::regex numberPattern(R"(^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$)");
    return std::regex_match(s, numberPattern);
}

ref<Expr> Executor::getParByIndex(std::tuple<int, int, int> item, ExecutionState *state, std::vector<ref<Expr>>& arguments) {
  ref<Expr> par;
  int firstIndex = std::get<0>(item);
  std::string arrayName = "_arg_"+llvm::utostr(firstIndex);
  int secondIndex = std::get<1>(item);
  if (secondIndex == -1){
    par = arguments[firstIndex];
  } else if (state->symbolicArrayMap.find(arrayName) != state->symbolicArrayMap.end()){
    SymArrayMemoryObject *symmo = state->symbolicArrayMap[arrayName];
    int thirdIndex = std::get<2>(item);
    if (thirdIndex == -1) {
      par = symmo->dimensionSize[secondIndex];
    } else {
      std::string elementName = arrayName+"["+llvm::utostr(secondIndex)+"]";
      if (state->symbolicArrayMap.find(elementName) != state->symbolicArrayMap.end()){
        SymArrayMemoryObject *element = state->symbolicArrayMap[elementName];
        par = element->dimensionSize[thirdIndex];
      }
    }
  }
  return par;
}

void Executor::runKernelFunction(Function *f, bool hasLaunchKernel) {
  KFunction *kf = kmodule->functionMap[f];
  assert(kf && "Entry function must exist in the KFunction map!");

  // Prepare the argument vector with symbolic variables
  std::vector<ref<Expr>> arguments;
  unsigned argIndex = 0;

  ExecutionState *state =
      new ExecutionState(kmodule->functionMap[f], memory.get());
  if (AllowSymLoop) {
    initializeLoopInfo(mainModule, *state);
  }

  // For each argument in the entry function, create a symbolic variable
  for (auto &arg : f->args()) {
      std::string symName = "_arg_" + std::to_string(argIndex);
      unsigned argSize = kmodule->targetData->getTypeAllocSize(arg.getType());
      Type *argType = arg.getType();
      bool isTensor = false;
      bool isTensorIterator = false;
      bool isScalar = false;
      bool isArrayRef = false;

      if (arg.getType()->isPointerTy()) {
        argSize = largeSymbolSize;
        argType = argType->getPointerElementType();
        llvm::StructType *structType = dyn_cast<llvm::StructType>(argType);
        if (structType && structType->getName().find("at::TensorIterator")!=std::string::npos) {
          isTensorIterator = true;
        } else if (structType && structType->getName().find("at::Tensor")!=std::string::npos) {
          isTensor = true;
        } else if (structType && structType->getName().find("c10::Scalar")!=std::string::npos){
          argSize = kmodule->targetData->getTypeAllocSize(argType);
          isScalar = true;
        } else if (structType && structType->getName().find("c10::ArrayRef")!=std::string::npos){
          isArrayRef = true;
        }
      }

      if (argumentValues.find(argIndex) != argumentValues.end()) {
        Interpreter::ParameterValue pval = argumentValues.at(argIndex);
        if (pval.type.find("int")!=std::string::npos && !pval.valueStr.empty() && isNumber(pval.valueStr)) {
          int64_t value = std::stoll(pval.valueStr);
          int sizeInBits = kmodule->targetData->getTypeSizeInBits(argType);
          ref<Expr> parExpr;
          if (sizeInBits == 1 || sizeInBits == 8 || sizeInBits == 16 || sizeInBits == 32 || sizeInBits == 64) {
            parExpr = ConstantExpr::create(value, sizeInBits, true);
          } else {
            parExpr = ConstantExpr::create(value, argSize*8, true);
            parExpr = ZExtExpr::create(parExpr, sizeInBits);
          }
          if (arg.getType()->isPointerTy()) {
            MemoryObject *pointerMo = memory->allocate(sizeInBits/8, /*isLocal=*/false, /*isGlobal=*/false,
                                    state, /*allocSite=*/state->pc->inst,
                                    /*alignment=*/8);  
            ObjectState *pointerOS = bindObjectInState(*state, pointerMo, false);
            pointerOS->write(0, parExpr); 
            parExpr = pointerMo->getBaseExpr();
          }
          arguments.push_back(parExpr);
          argIndex++;
          continue;
        } else if ((pval.type.find("float")!=std::string::npos || pval.type.find("double")!=std::string::npos) && !pval.valueStr.empty() && isNumber(pval.valueStr)) {
          ref<Expr> parExpr; 
          if (argSize == 4) {
            float f = std::stof(pval.valueStr);
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(float));
            parExpr = ConstantExpr::create(bits, argSize*8, true, true);
          } else {
            double d = std::stod(pval.valueStr);  
            uint64_t bits;
            std::memcpy(&bits, &d, sizeof(double));
            parExpr = ConstantExpr::create(bits, argSize*8, true, true);
          }
          arguments.push_back(parExpr);
          argIndex++;
          continue; 
        } else if (pval.type.find("bool")!=std::string::npos || pval.type.find("Bool")!=std::string::npos) {
          ref<Expr> parExpr;
          int sizeInBits = kmodule->targetData->getTypeSizeInBits(argType);
          bool isConstant = false;
          if (pval.valueStr.find("rue")!=std::string::npos) {
            parExpr = ConstantExpr::create(1, sizeInBits, false);
            isConstant = true;
          } else if (pval.valueStr.find("alse")!=std::string::npos){
            parExpr = ConstantExpr::create(0, sizeInBits, false);
            isConstant = true;
          }
          if (isConstant) {
            arguments.push_back(parExpr);
            argIndex++;
            continue; 
          }
        } else if (pval.type == "str") {
          bool isBasicStringPointer = false;
          if (arg.getType()->isPointerTy()) {
            StructType *structType = dyn_cast<StructType>(argType);
            if (structType->hasName()) {
              StringRef typeName = structType->getName();
              if (typeName.find("::basic_string")!=StringRef::npos) {
                isBasicStringPointer = true;
              }
            }
          }
          if (isBasicStringPointer) {
            bool isConstant = true;
            if (pval.valueStr.size() >= 2 && pval.valueStr[0] == 'u') {
              bool matchPattern = true;
              for (size_t i = 1; i < pval.valueStr.size(); ++i) {
                if (!std::isdigit(static_cast<unsigned char>(pval.valueStr[i]))) {
                  matchPattern = false;
                  break;
                }
              }
              if (matchPattern) {
                isConstant = false;
              }
            }
            if (isConstant) {
              size_t strSize = pval.valueStr.size() + 1; // include null terminator
              MemoryObject *moStr = memory->allocate(strSize, /*isLocal=*/false, /*isGlobal=*/false,
                                    state, /*allocSite=*/state->pc->inst,
                                    /*alignment=*/8);  
              ObjectState *osStr = bindObjectInState(*state, moStr, false);
              for (size_t i = 0; i < strSize - 1; ++i) {
                osStr->write(i, ConstantExpr::alloc(pval.valueStr[i], Expr::Int8));
              }
              osStr->write(strSize - 1, ConstantExpr::alloc('\0', Expr::Int8)); // null terminator

              argSize = kmodule->targetData->getTypeAllocSize(argType);
              MemoryObject *moObj =
                memory->allocate(argSize, /*isLocal=*/false, /*isGlobal=*/false,
                              state, /*allocSite=*/state->pc->inst,
                              /*alignment=*/8);    
              ObjectState *osObj = bindObjectInState(*state, moObj, false);
              // Write pointer to string data
              osObj->write(0, moStr->getBaseExpr());
              // Write size
              osObj->write(8, ConstantExpr::alloc(strSize-1, Expr::Int64));
              osObj->write(16, ConstantExpr::alloc(strSize-1, Expr::Int64)); // union.i64
              ref<Expr> parExpr = moObj->getBaseExpr();


              arguments.push_back(parExpr);
              argIndex++;
              continue; 
            }
          }
        } else if (pval.type.find("dtype")!=std::string::npos) {
          ref<Expr> dtype = getScalarTypeByStr(pval.valueStr);
          if (!dtype.isNull()) {
            if (arg.getType()->isPointerTy()) {
              int sizeInBits = kmodule->targetData->getTypeSizeInBits(argType);
              if (argType->isIntegerTy()) {
                MemoryObject *pointerMo = memory->allocate(sizeInBits/8, /*isLocal=*/false, /*isGlobal=*/false,
                                      state, /*allocSite=*/state->pc->inst,
                                      /*alignment=*/8);  
                ObjectState *pointerOS = bindObjectInState(*state, pointerMo, false);
                pointerOS->write(0, dtype); 
                ref<Expr> parExpr = pointerMo->getBaseExpr();
                arguments.push_back(parExpr);
                argIndex++;
                continue;
              }

              llvm::StructType *structType = dyn_cast<llvm::StructType>(argType);
              if (structType && structType->getName().find("optional")!=std::string::npos) {
                MemoryObject *pointerMo = memory->allocate(sizeInBits/8, /*isLocal=*/false, /*isGlobal=*/false,
                                      state, /*allocSite=*/state->pc->inst,
                                      /*alignment=*/8);  
                ObjectState *pointerOS = bindObjectInState(*state, pointerMo, false);
                pointerOS->write(0, dtype); 
                pointerOS->write(1, ConstantExpr::create(1, Expr::Int8)); 
                ref<Expr> parExpr = pointerMo->getBaseExpr();
                arguments.push_back(parExpr);
                argIndex++;
                continue;
              }
            }
          }
        } else if (pval.type.find("None")!=std::string::npos) {
          int sizeInBits = kmodule->targetData->getTypeSizeInBits(arg.getType());
          arguments.push_back(ConstantExpr::create(0, sizeInBits, false));
          argIndex++;
          continue; 
        } else if (pval.type.find("list")!=std::string::npos) {
          MemoryObject *mo =
          memory->allocate(argSize, /*isLocal=*/false, /*isGlobal=*/false,
                          state, /*allocSite=*/state->pc->inst,
                          /*alignment=*/8);    
          mo->setName(symName);
          const Array *array = arrayCache.CreateArray(symName, mo->size, false);
          bindObjectInState(*state, mo, false, array);
          state->addSymbolic(mo, array);

          SymArrayMemoryObject *symmo;
          int len = -1;
          if (!pval.shape.empty() && pval.shape.find(0)!=pval.shape.end() && pval.shape[0] >= 0) {
            len = pval.shape[0];
          } else if (pval.dim >= 0) {
            len = pval.dim;
          }
          if (len >= 0) {
            int size = 0;
            if (argType->isIntegerTy() || argType->isFloatTy() || argType->isDoubleTy() || argType->isHalfTy() || argType->isBFloatTy()) {
              uint64_t elementSize = kmodule->targetData->getTypeAllocSize(argType);
              size = len * elementSize;
            } else if (argType->isArrayTy()) {
              llvm::Type *elementTy = argType->getArrayElementType();
              uint64_t elementSize = kmodule->targetData->getTypeAllocSize(elementTy);
              size = len * elementSize;
            } else {
              size = len;
            }
            ref<Expr> sizeExpr = ConstantExpr::create(size, Expr::Int64, false);
            symmo = new SymArrayMemoryObject(mo->address, array, nullptr, sizeExpr);
          } else {
            auto pair = createSizeSymbol(*state, symName + ".size", true);
            symmo = new SymArrayMemoryObject(mo->address, array, pair.first, pair.second);
          }
          state->symbolicArrayMap[symName] = symmo;

          if (pval.elementType.find("tensor")!=std::string::npos || pval.elementType.find("Tensor")!=std::string::npos) {
            int i = 0;
            for (auto &a: pval.content) {
              MemoryObject *tensorMo = loadTensorConfig(state, a, symName + "[" + std::to_string(i) + "]", argSize);
              if (tensorMo==nullptr) {
                klee_error("not support unknown dim tensor in list");
              }
              symmo->constantIndexAddress[i] = tensorMo->getBaseExpr();
              i+=1;
            }
          }

          if (pval.maxVal < INT32_MAX) {
            symmo->maxVal = ConstantExpr::create(pval.maxVal, Expr::Int32, true);
          }
          if (pval.minVal > INT32_MIN) {
            symmo->minVal = ConstantExpr::create(pval.minVal, Expr::Int32, true);
          }
          symmo->hasDuplicateVal = pval.hasDuplicateVal;

          ref<Expr> parExpr = mo->getBaseExpr();
          arguments.push_back(parExpr);
          argIndex++;
          continue;
        } else if (pval.type.find("TensorIterator")!=std::string::npos) {
          MemoryObject *mo =
            memory->allocate(argSize, /*isLocal=*/false, /*isGlobal=*/false,
                            state, /*allocSite=*/state->pc->inst,
                            /*alignment=*/8);    
          mo->setName(symName);
          const Array *array = arrayCache.CreateArray(symName, mo->size, false);
          bindObjectInState(*state, mo, false, array);
          state->addSymbolic(mo, array);

          TensorIterator *iterator = new TensorIterator(mo->address, array, nullptr, nullptr);
          state->inputIterator = iterator;

          if (pval.dim != 0) {
            iterator->size = ConstantExpr::create(pval.dim, Expr::Int64, false);
          } else {
            auto pair = createSizeSymbol(*state, symName + ".size", true);
            addConstraint(*state, UgeExpr::create(pair.second, ConstantExpr::create(1, pair.second->getWidth())));
            iterator->sizeName = pair.first;
            iterator->size = pair.second;
          }

          int outputs = 0;
          int i = 0;
          for (auto &a: pval.content) {
            MemoryObject *tensorMo = loadTensorConfig(state, a, symName + "_tensor_" + std::to_string(i), argSize);
            if (tensorMo==nullptr) {
              klee_error("not support unknown dim tensor in iterator");
            }
            iterator->addTensor(tensorMo->name, a.isConstTensor);
            i+=1;

            if (a.isOutput) outputs+=1;
          }
          iterator->noutputs = outputs;

          ref<Expr> parExpr = mo->getBaseExpr();
          arguments.push_back(parExpr);
          argIndex++;
          continue;
        } else if (pval.type.find("tensor")!=std::string::npos || pval.type.find("Tensor")!=std::string::npos) {
          MemoryObject *mo = loadTensorConfig(state, pval, symName, argSize);
          ref<Expr> parExpr = mo->getBaseExpr();
          arguments.push_back(parExpr);
          argIndex++;
          continue;
        }
      }

      MemoryObject *mo =
            memory->allocate(argSize, /*isLocal=*/false, /*isGlobal=*/false,
                            state, /*allocSite=*/state->pc->inst,
                            /*alignment=*/8);    
      if (!mo)
        klee_error("Could not allocate memory for function arguments");
      
      if (isScalar) {
        ObjectState *os = bindObjectInState(*state, mo, false);

        std::string typeName = symName + ".tag";
        MemoryObject *typeMo =
              memory->allocate(4, /*isLocal=*/false, /*isGlobal=*/false,
                              state, /*allocSite=*/state->prevPC->inst,
                              /*alignment=*/8);          
        typeMo->setName(typeName);
        const Array *typeArray = arrayCache.CreateArray(typeName, typeMo->size, true);
        bindObjectInState(*state, typeMo, false, typeArray);
        state->addSymbolic(typeMo, typeArray);
        ref<Expr> scalarTypeExpr = Expr::createTempRead(typeArray, typeMo->size*8);
        addConstraint(*state, UgeExpr::create(scalarTypeExpr, ConstantExpr::create(0, scalarTypeExpr->getWidth())));
        addConstraint(*state, UleExpr::create(scalarTypeExpr, ConstantExpr::create(4, scalarTypeExpr->getWidth()))); // not support symoblic value now
        os->write(0, scalarTypeExpr);

        std::string double1Name = symName + "_double1";
        MemoryObject *d1Mo =
          memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                          state, /*allocSite=*/state->prevPC->inst,
                          /*alignment=*/8);          
        ref<Expr> d1Expr = symbolizeIndex(*state, double1Name, d1Mo, true);
        os->write(16, d1Expr);

        std::string double2Name = symName + "_double2";
        MemoryObject *d2Mo =
          memory->allocate(8, /*isLocal=*/false, /*isGlobal=*/false,
                          state, /*allocSite=*/state->prevPC->inst,
                          /*alignment=*/8);          
        ref<Expr> d2Expr = symbolizeIndex(*state, double2Name, d2Mo, true);
        os->write(24, d1Expr);

        ref<Expr> parExpr = mo->getBaseExpr();
        arguments.push_back(parExpr);
        argIndex++;
        continue;
      }
      
      mo->setName(symName);
      bool isIntVar = arg.getType()->isIntegerTy() || arg.getType()->isFloatingPointTy();
      const Array *array = arrayCache.CreateArray(symName, mo->size, isIntVar);
      bindObjectInState(*state, mo, false, array);
      state->addSymbolic(mo, array);

      ref<Expr> parExpr;
      if (arg.getType()->isPointerTy()) {
        if (isTensorIterator) {
          auto pair = createSizeSymbol(*state, symName + ".size", true);
          ref<Expr> sizeExpr = pair.second;
          addConstraint(*state, UgeExpr::create(sizeExpr, ConstantExpr::create(1, sizeExpr->getWidth())));
          TensorIterator *iterator = new TensorIterator(mo->address, array, pair.first, sizeExpr);
          // for test
          iterator->noutputs = 1;
          iterator->size = ConstantExpr::create(3, Expr::Int32);
          
          for(int i = 0;i<3;i++) {

            Interpreter::ParameterValue p("torch.tensor", "", i+2, "");
            MemoryObject *tensorMo = loadTensorConfig(state, p, symName + "_tensor_" + std::to_string(i), argSize);
            iterator->addTensor(tensorMo->name, false);


          }

          state->inputIterator = iterator;
        } else {
          auto pair = createSizeSymbol(*state, symName + ".size", true);
          SymArrayMemoryObject *symmo = new SymArrayMemoryObject(mo->address, array, pair.first, pair.second);
          state->symbolicArrayMap[symName] = symmo;

          std::string dimName = symName + ".dim";
          auto dimpair = createSizeSymbol(*state, dimName);
          ref<Expr> dimExpr = dimpair.second; 
          symmo->shapeSize = dimExpr;

          if (isTensor) {
            std::string elementSizeName = symName + ".item_size";
            auto pair2 = createSizeSymbol(*state, elementSizeName);
            ref<Expr> elementSizeExpr = pair2.second;
            symmo->elementSize = elementSizeExpr;
            addConstraint(*state, UgeExpr::create(elementSizeExpr, ConstantExpr::create(1, elementSizeExpr->getWidth())));
            addConstraint(*state, UleExpr::create(elementSizeExpr, ConstantExpr::create(16, elementSizeExpr->getWidth())));

            std::map<int, ref<Expr>> dimensionSize;
            if (argumentValues.find(argIndex) != argumentValues.end()) {
              Interpreter::ParameterValue pval = argumentValues.at(argIndex);
              if (!pval.shape.empty()) {
                for (auto pair: pval.shape) {
                  dimensionSize[pair.first] = ConstantExpr::create(pair.second, Expr::Int64, false);
                }
                symmo->dimensionSize = dimensionSize;
              }
            }
          }
          if (isArrayRef) {
            MemoryObject *argMo =
              memory->allocate(16, /*isLocal=*/false, /*isGlobal=*/false,
                              state, /*allocSite=*/state->pc->inst,
                              /*alignment=*/8); 
            ObjectState *argOS = bindObjectInState(*state, argMo, false);
            argOS->write(0, mo->getBaseExpr());
            argOS->write(8, ZExtExpr::create(pair.second, Expr::Int64));
            mo = argMo;
          }
        }
        parExpr = mo->getBaseExpr();
      } else {
        int sizeInBits = kmodule->targetData->getTypeSizeInBits(argType);
        if (sizeInBits == 1 || sizeInBits == 8 || sizeInBits == 16 || sizeInBits == 32 || sizeInBits == 64) {
          parExpr = Expr::createTempRead(array, sizeInBits);
        } else {
          parExpr = Expr::createTempRead(array, argSize*8);
          parExpr = ZExtExpr::create(parExpr, sizeInBits);
        }
        
        if (isIntVar) {
          state->intArgAddBound[array->name] = std::make_pair(parExpr, false);
          if (parExpr->getWidth() <= 64) {
            ref<ConstantExpr> minValueExpr = ConstantExpr::alloc(1ULL << (parExpr->getWidth() - 1), parExpr->getWidth(), true);   
            addConstraint(*state, SgeExpr::create(parExpr, minValueExpr)); 
            ref<ConstantExpr> maxValueExpr;
            if (parExpr->getWidth() < 64) {
              maxValueExpr = ConstantExpr::create((1ULL << parExpr->getWidth()) - 1, parExpr->getWidth());      
            } else if (parExpr->getWidth() == 64){
              maxValueExpr = ConstantExpr::alloc(18446744073709551615ULL, 64);
            }
            addConstraint(*state, UleExpr::create(parExpr, maxValueExpr));
          } 
        }
      }
      arguments.push_back(parExpr);
      argIndex++;
  }

  auto batchSizeSymbol = createSizeSymbol(*state, "batch_size");
  ref<Expr> batchSizeExpr = batchSizeSymbol.second;
  addConstraint(*state, UgeExpr::create(batchSizeExpr, ConstantExpr::create(1, batchSizeExpr->getWidth())));
  if (promptCons.find("batch_size") != promptCons.end()) {
    addConstraint(*state, UleExpr::create(batchSizeExpr, ConstantExpr::create(promptCons["batch_size"], batchSizeExpr->getWidth())));
  } else {
    addConstraint(*state, UleExpr::create(batchSizeExpr, ConstantExpr::create(1000, batchSizeExpr->getWidth())));
  }
  auto seqLenSymbol = createSizeSymbol(*state, "seq_len");
  ref<Expr> seqLenExpr = seqLenSymbol.second;
  addConstraint(*state, UgeExpr::create(seqLenExpr, ConstantExpr::create(1, seqLenExpr->getWidth())));
  if (promptCons.find("seq_len") != promptCons.end()) {
    addConstraint(*state, UleExpr::create(seqLenExpr, ConstantExpr::create(promptCons["seq_len"], seqLenExpr->getWidth())));
  } else {
    addConstraint(*state, UleExpr::create(seqLenExpr, ConstantExpr::create(2000000, seqLenExpr->getWidth())));
  }
  if (promptCons.find("num_tokens") != promptCons.end()) {
    addConstraint(*state, UleExpr::create(MulExpr::create(batchSizeExpr, seqLenExpr), ConstantExpr::create(promptCons["num_tokens"], seqLenExpr->getWidth())));
  } else {
    addConstraint(*state, UleExpr::create(MulExpr::create(batchSizeExpr, seqLenExpr), ConstantExpr::create(9000000, seqLenExpr->getWidth())));
  }

  for (auto &pcon: parCons) {
    std::string mathExpr = pcon.first;
    ref<Expr> expr = nullptr;
    bool isInt = true;

    std::stringstream ss(mathExpr);
    std::string term;
    while (std::getline(ss, term, '+')) {
      int divPos = term.find('/');
      double dividend = 1.0;
      std::string numPart = term;
      if (divPos != std::string::npos) {
          std::string denStr = term.substr(divPos + 1);
          numPart = term.substr(0, divPos);

          std::stringstream iss(denStr);
          iss >> dividend;
      }

      if (term.find("b*s") != std::string::npos || term.find("s*b") != std::string::npos) {
          double bs_co = parseCoefficient(term);
          if (std::floor(bs_co) == bs_co) {
            int n = static_cast<int>(bs_co);
            if (n) {
              ref<Expr> tmpExpr = MulExpr::create(batchSizeExpr, seqLenExpr);
              tmpExpr = MulExpr::create(tmpExpr, ConstantExpr::create(n, tmpExpr->getWidth(), true));
              if (expr) {
                expr = AddExpr::create(expr, ZExtExpr::create(tmpExpr, expr->getWidth()));
              } else {
                expr = tmpExpr;
              }
            }
          } else {
            isInt = false;
            break;
          }
          if (std::floor(dividend) == dividend) {
            if (dividend != 1.0) {
              int n = static_cast<int>(dividend);
              if (n) {
                expr = SDivExpr::create(expr, ConstantExpr::create(n, expr->getWidth(), true));
              }
            }
          } else {
            isInt = false;
            break;
          }
      } else if (term.find("*b") != std::string::npos) {
          double b_co = parseCoefficient(term);
          if (std::floor(b_co) == b_co) {
            int n = static_cast<int>(b_co);
            if (n) {
              ref<Expr> tmpExpr = MulExpr::create(batchSizeExpr, ConstantExpr::create(n, batchSizeExpr->getWidth(), true));
              if (expr) {
                expr = AddExpr::create(expr, ZExtExpr::create(tmpExpr, expr->getWidth()));
              } else {
                expr = tmpExpr;
              }
            }
          } else {
            isInt = false;
            break;
          }
          if (std::floor(dividend) == dividend) {
            if (dividend != 1.0) {
              int n = static_cast<int>(dividend);
              if (n) {
                expr = SDivExpr::create(expr, ConstantExpr::create(n, expr->getWidth(), true));
              }
            }
          } else {
            isInt = false;
            break;
          }
      } else if (term.find("*s") != std::string::npos) {
          double s_co =  parseCoefficient(term);
          if (std::floor(s_co) == s_co) {
            int n = static_cast<int>(s_co);
            if (n) {
              ref<Expr> tmpExpr = MulExpr::create(seqLenExpr, ConstantExpr::create(n, seqLenExpr->getWidth(), true));
              if (expr) {
                expr = AddExpr::create(expr, ZExtExpr::create(tmpExpr, expr->getWidth()));
              } else {
                expr = tmpExpr;
              }
            }
          } else {
            isInt = false;
            break;
          }
          if (std::floor(dividend) == dividend) {
            if (dividend != 1.0) {
              int n = static_cast<int>(dividend);
              if (n) {
                expr = SDivExpr::create(expr, ConstantExpr::create(n, expr->getWidth(), true));
              }
            }
          } else {
            isInt = false;
            break;
          }
      } else {
        double num;
        std::istringstream iss(numPart);
        iss >> num;
        if (!iss.fail() && iss.eof()) {
          if (std::floor(num) == num && dividend == 1.0) {
            int n = static_cast<int>(num);
            if (expr) {
              expr = AddExpr::create(expr, ConstantExpr::create(n, expr->getWidth(), true));
            } else {
              expr = ConstantExpr::create(n, Expr::Int64, true);
            }
          } else if (std::floor(num) == num && std::floor(dividend) == dividend) {
            int n = static_cast<int>(num);
            int d = static_cast<int>(dividend);
            ref<Expr> tmpExpr = SDivExpr::create(ConstantExpr::create(n, Expr::Int64, true), ConstantExpr::create(d, Expr::Int64, true));
            if (expr) {
              expr = AddExpr::create(expr, SExtExpr::create(tmpExpr, expr->getWidth()));
            } else {
              expr = tmpExpr;
            }
          } else {
            isInt = false;
            break;
          }
        } else {
          isInt = false;
          break;
        }
      }
    }

    std::vector<std::tuple<int, int, int>> equals = pcon.second;
    if (!symRangeCons.empty() && symRangeCons.find(mathExpr) != symRangeCons.end()) {
      int firstIndex = std::get<0>(equals[0]);
      if (firstIndex >= 0) {
        ref<Expr> left = getParByIndex(equals[0], state, arguments);
        int symMin = symRangeCons[mathExpr].first;
        int symMax = symRangeCons[mathExpr].second;
        ref<Expr> minExpr = SgeExpr::create(left, ConstantExpr::create(symMin, left->getWidth()));
        ref<Expr> maxExpr = SleExpr::create(left, ConstantExpr::create(symMax, left->getWidth()));
        addConstraint(*state, minExpr);
        addConstraint(*state, maxExpr);
      }
    }

    if (expr && isInt) {
      for (size_t i = 0; i < equals.size(); ++i) {
        int firstIndex = std::get<0>(equals[i]);
        if (firstIndex < 0) {
          int secondIndex = std::get<1>(equals[i]);
          std::string arrayName = "_arg_"+llvm::utostr(secondIndex);
          int thirdIndex = std::get<2>(equals[i]);
          if (thirdIndex >= 0) {
            arrayName += "[" + std::to_string(thirdIndex) + "]";
          }
          if (state->symbolicArrayMap.find(arrayName) != state->symbolicArrayMap.end()){
            SymArrayMemoryObject *symmo = state->symbolicArrayMap[arrayName];
            if (firstIndex == -1) {
              symmo->maxVal = expr;
            } else if (firstIndex == -2) {
              symmo->minVal = expr;
            } 
          } else {
            klee_warning("max/min value for invalid array");
          }
        } else {
          ref<Expr> left = getParByIndex(equals[i], state, arguments);
          addConstraint(*state, EqExpr::create(left, ZExtExpr::create(expr, left->getWidth())));
        }
      }
    } else {
      if (equals.size() < 2) {
        continue;
      }

      int leftIndex = 0;
      int firstIndex = std::get<0>(equals[leftIndex]);
      while(firstIndex < 0) {
        leftIndex+=1;
        firstIndex = std::get<0>(equals[leftIndex]);
      }
      ref<Expr> left = getParByIndex(equals[leftIndex], state, arguments);
      for (int i = leftIndex + 1; i < equals.size(); ++i) {
        int firstIndex = std::get<0>(equals[i]);
        if (firstIndex < 0) {
          int secondIndex = std::get<1>(equals[i]);
          std::string arrayName = "_arg_"+llvm::utostr(secondIndex);
          int thirdIndex = std::get<2>(equals[i]);
          if (thirdIndex >= 0) {
            arrayName += "[" + std::to_string(thirdIndex) + "]";
          }
          if (state->symbolicArrayMap.find(arrayName) != state->symbolicArrayMap.end()){
            SymArrayMemoryObject *symmo = state->symbolicArrayMap[arrayName];
            if (firstIndex == -1) {
              symmo->maxVal = left;
            } else if (firstIndex == -2) {
              symmo->minVal = left;
            } 
          } else {
            klee_warning("max/min value for invalid array");
          }
        } else {
          ref<Expr> right = getParByIndex(equals[i], state, arguments);
          if (!isa<ConstantExpr>(left) && !isa<ConstantExpr>(right)) {
            addConstraint(*state, EqExpr::create(left, ZExtExpr::create(right, left->getWidth())));
          }
        }
      }

      for (int i = 0; i < leftIndex; ++i) {
        int firstIndex = std::get<0>(equals[i]);
        if (firstIndex < 0) {
          int secondIndex = std::get<1>(equals[i]);
          std::string arrayName = "_arg_"+llvm::utostr(secondIndex);
          int thirdIndex = std::get<2>(equals[i]);
          if (thirdIndex >= 0) {
            arrayName += "[" + std::to_string(thirdIndex) + "]";
          }
          if (state->symbolicArrayMap.find(arrayName) != state->symbolicArrayMap.end()){
            SymArrayMemoryObject *symmo = state->symbolicArrayMap[arrayName];
            if (firstIndex == -1) {
              symmo->maxVal = left;
            } else if (firstIndex == -2) {
              symmo->minVal = left;
            } 
          } else {
            klee_warning("max/min value for invalid array");
          }
        }
      }
    }
  }

  if (pathWriter) 
    state->pathOS = pathWriter->open();
  if (symPathWriter) 
    state->symPathOS = symPathWriter->open();

  if (statsTracker)
    statsTracker->framePushed(*state, 0);

  // Ensure the argument count matches the function's signature
  assert(arguments.size() == f->arg_size() && "Mismatch in number of arguments!");
  for (unsigned i = 0, e = f->arg_size(); i != e; ++i)
    bindArgument(kf, i, *state, arguments[i]);
  
  if (!hasLaunchKernel){
    const char *kernelConfigArr[7] = {
        "gridDim.x",
        "gridDim.y",
        "gridDim.z",
        "blockDim.x",
        "blockDim.y",
        "blockDim.z",
        "sharedMemSize"
    };
    ref<Expr> configExprs[7];

    for (int i = 0; i < 7; i++) {
        MemoryObject *mo = memory->allocate(sizeof(unsigned), false, /*isGlobal=*/false, state, state->pc->inst, 4);    
        configExprs[i] = symbolizeIndex(*state, kernelConfigArr[i], mo);
    }
    initializeCudaKernelConfig(*state, configExprs[0], configExprs[1], configExprs[2], configExprs[3], configExprs[4], configExprs[5], configExprs[6]);
  }
  
  initializeGlobals(*state);

  executionTree = createExecutionTree(
      *state, userSearcherRequiresInMemoryExecutionTree(), *interpreterHandler);
  run(*state);
  executionTree = nullptr;

  // hack to clear memory objects
  memory = nullptr;

  isIntArgSigned.clear();
  globalObjects.clear();
  globalAddresses.clear();
  sharedAddresses.clear();
  deletedAddresses.clear();
  mainModule = nullptr;

  if (statsTracker)
    statsTracker->done();
}

/***/

void Executor::runFunctionAsMain(Function *f,
				 int argc,
				 char **argv,
				 char **envp) {
  std::vector<ref<Expr> > arguments;

  // force deterministic initialization of memory objects
  srand(1);
  srandom(1);
  
  MemoryObject *argvMO = 0;

  // In order to make uclibc happy and be closer to what the system is
  // doing we lay out the environments at the end of the argv array
  // null that uclibc seems to expect, possibly the ELF header?

  int envc;
  for (envc=0; envp[envc]; ++envc) ;

  unsigned NumPtrBytes = Context::get().getPointerWidth() / 8;
  KFunction *kf = kmodule->functionMap[f];
  assert(kf);
  Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
  if (ai!=ae) {
    arguments.push_back(ConstantExpr::alloc(argc, Expr::Int32));
    if (++ai!=ae) {
      Instruction *first = &*(f->begin()->begin());
      argvMO = memory->allocate((argc + 1 + envc + 1 + 1) * NumPtrBytes,
                                /*isLocal=*/false, /*isGlobal=*/true,
                                /*state=*/nullptr, /*allocSite=*/first,
                                /*alignment=*/8);

      if (!argvMO)
        klee_error("Could not allocate memory for function arguments");

      arguments.push_back(argvMO->getBaseExpr());

      if (++ai!=ae) {
        uint64_t envp_start = argvMO->address + (argc+1)*NumPtrBytes;
        arguments.push_back(Expr::createPointer(envp_start));

        if (++ai!=ae)
          klee_error("invalid main function (expect 0-3 arguments)");
      }
    }
  }

  ExecutionState *state =
      new ExecutionState(kmodule->functionMap[f], memory.get());

  if (AllowSymLoop) {
    initializeLoopInfo(mainModule, *state);
  }
  if (pathWriter) 
    state->pathOS = pathWriter->open();
  if (symPathWriter) 
    state->symPathOS = symPathWriter->open();


  if (statsTracker)
    statsTracker->framePushed(*state, 0);

  assert(arguments.size() == f->arg_size() && "wrong number of arguments");
  for (unsigned i = 0, e = f->arg_size(); i != e; ++i)
    bindArgument(kf, i, *state, arguments[i]);

  if (argvMO) {
    ObjectState *argvOS = bindObjectInState(*state, argvMO, false);

    for (int i=0; i<argc+1+envc+1+1; i++) {
      if (i==argc || i>=argc+1+envc) {
        // Write NULL pointer
        argvOS->write(i * NumPtrBytes, Expr::createPointer(0));
      } else {
        char *s = i<argc ? argv[i] : envp[i-(argc+1)];
        int j, len = strlen(s);

        MemoryObject *arg =
            memory->allocate(len + 1, /*isLocal=*/false, /*isGlobal=*/true,
                             state, /*allocSite=*/state->pc->inst,
                             /*alignment=*/8);
        if (!arg)
          klee_error("Could not allocate memory for function arguments");
        ObjectState *os = bindObjectInState(*state, arg, false);
        for (j=0; j<len+1; j++)
          os->write8(j, s[j]);

        // Write pointer to newly allocated and initialised argv/envp c-string
        argvOS->write(i * NumPtrBytes, arg->getBaseExpr());
      }
    }
  }
  
  initializeGlobals(*state);

  executionTree = createExecutionTree(
      *state, userSearcherRequiresInMemoryExecutionTree(), *interpreterHandler);
  run(*state);
  executionTree = nullptr;

  // hack to clear memory objects
  memory = nullptr;

  globalObjects.clear();
  globalAddresses.clear();

  if (statsTracker)
    statsTracker->done();
}

unsigned Executor::getPathStreamID(const ExecutionState &state) {
  assert(pathWriter);
  return state.pathOS.getID();
}

unsigned Executor::getSymbolicPathStreamID(const ExecutionState &state) {
  assert(symPathWriter);
  return state.symPathOS.getID();
}

void Executor::getConstraintLog(const ExecutionState &state, std::string &res,
                                Interpreter::LogType logFormat) {

  switch (logFormat) {
  case STP: {
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool), state.intArrNames, state.getIOExprs());
    res = solver->getConstraintLog(query);
  } break;

  case KQUERY: {
    std::string Str;
    llvm::raw_string_ostream info(Str);
    ExprPPrinter::printConstraints(info, state.constraints);
    res = info.str();
  } break;

  case SMTLIB2: {
    std::string Str;
    llvm::raw_string_ostream info(Str);
    ExprSMTLIBPrinter printer;
    printer.setOutput(info);
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool), state.intArrNames, state.getIOExprs());
    printer.setQuery(query);
    printer.generateOutput();
    res = info.str();
  } break;

  default:
    klee_warning("Executor::getConstraintLog() : Log format not supported!");
  }
}

bool Executor::getSymbolicSolution(const ExecutionState &state,
                                   std::vector< 
                                   std::pair<std::string,
                                   std::vector<unsigned char> > >
                                   &res) {
  solver->setTimeout(coreSolverTimeout);

  ConstraintSet extendedConstraints(state.constraints);
  ConstraintManager cm(extendedConstraints);

  // Go through each byte in every test case and attempt to restrict
  // and therefore human readable. It is also possible to customize
  // the preferred constraints.  See test/Features/PreferCex.c for
  // also make understanding individual test cases much easier.
  for (auto& pi: state.cexPreferences) {
    bool mustBeTrue;
    // Attempt to bound byte to constraints held in cexPreferences
    bool success =
      solver->mustBeTrue(extendedConstraints, Expr::createIsZero(pi),
        mustBeTrue, state.queryMetaData, state.intArrNames, state.getIOExprs());
    // If it isn't possible to add the condition without making the entire list
    // UNSAT, then just continue to the next condition
    if (!success) break;
    // If the particular constraint operated on in this iteration through
    // the loop isn't implied then add it to the list of constraints.
    if (!mustBeTrue)
      cm.addConstraint(pi);
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);
  bool success = solver->getInitialValues(extendedConstraints, objects, values,
                                          state.queryMetaData);
  solver->setTimeout(time::Span());
  if (!success) {
    klee_warning("unable to compute initial values (invalid constraints?)!");
    ExprPPrinter::printQuery(llvm::errs(), state.constraints,
                             ConstantExpr::alloc(0, Expr::Bool));
    return false;
  }
  
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    res.push_back(std::make_pair(state.symbolics[i].first->name, values[i]));
  return true;
}

void Executor::getCoveredLines(const ExecutionState &state,
                               std::map<const std::string*, std::set<unsigned> > &res) {
  res = state.coveredLines;
}

void Executor::doImpliedValueConcretization(ExecutionState &state,
                                            ref<Expr> e,
                                            ref<ConstantExpr> value) {
  abort(); // FIXME: Broken until we sort out how to do the write back.

  if (DebugCheckForImpliedValues)
    ImpliedValue::checkForImpliedValues(solver->solver.get(), e, value);

  ImpliedValueList results;
  ImpliedValue::getImpliedValues(e, value, results);
  for (ImpliedValueList::iterator it = results.begin(), ie = results.end();
       it != ie; ++it) {
    ReadExpr *re = it->first.get();
    
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(re->index)) {
      // FIXME: This is the sole remaining usage of the Array object
      // variable. Kill me.
      const MemoryObject *mo = 0; //re->updates.root->object;
      const ObjectState *os = state.addressSpace.findObject(mo);

      if (!os) {
        // in other cases we would like to concretize the outstanding
      } else {
        assert(!os->readOnly && 
               "not possible? read only object with static read?");
        ObjectState *wos = state.addressSpace.getWriteable(mo, os);
        wos->write(CE, it->second);
      }
    }
  }
}

Expr::Width Executor::getWidthForLLVMType(llvm::Type *type) const {
  return kmodule->targetData->getTypeSizeInBits(type);
}

size_t Executor::getAllocationAlignment(const llvm::Value *allocSite) const {
  // FIXME: 8 was the previous default. We shouldn't hard code this
  // and should fetch the default from elsewhere.
  const size_t forcedAlignment = 8;
  size_t alignment = 0;
  llvm::Type *type = NULL;
  std::string allocationSiteName(allocSite->getName().str());
  if (const GlobalObject *GO = dyn_cast<GlobalObject>(allocSite)) {
    alignment = GO->getAlignment();
    if (const GlobalVariable *globalVar = dyn_cast<GlobalVariable>(GO)) {
      // All GlobalVariables's have pointer type
      assert(globalVar->getType()->isPointerTy() &&
             "globalVar's type is not a pointer");
      type = globalVar->getValueType();
    } else {
      type = GO->getType();
    }
  } else if (const AllocaInst *AI = dyn_cast<AllocaInst>(allocSite)) {
    alignment = AI->getAlign().value();
    type = AI->getAllocatedType();
  } else if (isa<InvokeInst>(allocSite) || isa<CallInst>(allocSite)) {
    // FIXME: Model the semantics of the call to use the right alignment
    const CallBase &cb = cast<CallBase>(*allocSite);
    llvm::Function *fn =
        klee::getDirectCallTarget(cb, /*moduleIsFullyLinked=*/true);
    if (fn)
      allocationSiteName = fn->getName().str();

    klee_warning_once(fn != NULL ? fn : allocSite,
                      "Alignment of memory from call \"%s\" is not "
                      "modelled. Using alignment of %zu.",
                      allocationSiteName.c_str(), forcedAlignment);
    alignment = forcedAlignment;
  } else {
    llvm_unreachable("Unhandled allocation site");
  }

  if (alignment == 0) {
    assert(type != NULL);
    // No specified alignment. Get the alignment for the type.
    if (type->isSized()) {
#if LLVM_VERSION_CODE >= LLVM_VERSION(16, 0)
      alignment = kmodule->targetData->getPrefTypeAlign(type).value();
#else
      alignment = kmodule->targetData->getPrefTypeAlignment(type);
#endif

    } else {
      klee_warning_once(allocSite, "Cannot determine memory alignment for "
                                   "\"%s\". Using alignment of %zu.",
                        allocationSiteName.c_str(), forcedAlignment);
      alignment = forcedAlignment;
    }
  }

  // Currently we require alignment be a power of 2
  if (!bits64::isPowerOfTwo(alignment)) {
    klee_warning_once(allocSite, "Alignment of %zu requested for %s but this "
                                 "not supported. Using alignment of %zu",
                      alignment, allocSite->getName().str().c_str(),
                      forcedAlignment);
    alignment = forcedAlignment;
  }
  assert(bits64::isPowerOfTwo(alignment) &&
         "Returned alignment must be a power of two");
  return alignment;
}

void Executor::prepareForEarlyExit() {
  if (statsTracker) {
    // Make sure stats get flushed out
    statsTracker->done();
  }
}

/// Returns the errno location in memory
int *Executor::getErrnoLocation(const ExecutionState &state) const {
#if !defined(__APPLE__) && !defined(__FreeBSD__)
  /* From /usr/include/errno.h: it [errno] is a per-thread variable. */
  return __errno_location();
#else
  return __error();
#endif
}

void Executor::dumpExecutionTree() {
  if (!::dumpExecutionTree)
    return;

  char name[32];
  snprintf(name, sizeof(name), "exec_tree%08d.dot", (int)stats::instructions);
  auto os = interpreterHandler->openOutputFile(name);
  if (os)
    executionTree->dump(*os);

  ::dumpExecutionTree = 0;
}

void Executor::dumpStates() {
  if (!::dumpStates) return;

  auto os = interpreterHandler->openOutputFile("states.txt");

  if (os) {
    for (ExecutionState *es : states) {
      *os << "(" << es << ",";
      *os << "[";
      auto next = es->stack.begin();
      ++next;
      for (auto sfIt = es->stack.begin(), sf_ie = es->stack.end();
           sfIt != sf_ie; ++sfIt) {
        *os << "('" << sfIt->kf->function->getName().str() << "',";
        if (next == es->stack.end()) {
          *os << es->prevPC->info->line << "), ";
        } else {
          *os << next->caller->info->line << "), ";
          ++next;
        }
      }
      *os << "], ";

      StackFrame &sf = es->stack.back();
      uint64_t md2u = computeMinDistToUncovered(es->pc,
                                                sf.minDistToUncoveredOnReturn);
      uint64_t icnt = theStatisticManager->getIndexedValue(stats::instructions,
                                                           es->pc->info->id);
      uint64_t cpicnt = sf.callPathNode->statistics.getValue(stats::instructions);

      *os << "{";
      *os << "'depth' : " << es->depth << ", ";
      *os << "'queryCost' : " << es->queryMetaData.queryCost << ", ";
      *os << "'coveredNew' : " << es->coveredNew << ", ";
      *os << "'instsSinceCovNew' : " << es->instsSinceCovNew << ", ";
      *os << "'md2u' : " << md2u << ", ";
      *os << "'icnt' : " << icnt << ", ";
      *os << "'CPicnt' : " << cpicnt << ", ";
      *os << "}";
      *os << ")\n";
    }
  }

  ::dumpStates = 0;
}

///

Interpreter *Interpreter::create(LLVMContext &ctx, const InterpreterOptions &opts,
                                 InterpreterHandler *ih) {
  return new Executor(ctx, opts, ih);
}
