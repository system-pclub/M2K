//===-- SpecialFunctionHandler.cpp ----------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SpecialFunctionHandler.h"

#include "ExecutionState.h"
#include "Executor.h"
#include "Memory.h"
#include "MemoryManager.h"
#include "MergeHandler.h"
#include "Searcher.h"
#include "StatsTracker.h"
#include "TimingSolver.h"

#include "klee/Config/config.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"
#include "klee/Solver/SolverCmdLine.h"
#include "klee/Support/Casting.h"
#include "klee/Support/Debug.h"
#include "klee/Support/ErrorHandling.h"
#include "klee/Support/OptionCategories.h"

#include "klee/Support/CompilerWarning.h"
DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
DISABLE_WARNING_POP

#include <array>
#include <cerrno>
#include <sstream>

using namespace llvm;
using namespace klee;

namespace {
cl::opt<bool>
    ReadablePosix("readable-posix-inputs", cl::init(false),
                  cl::desc("Prefer creation of POSIX inputs (command-line "
                           "arguments, files, etc.) with human readable bytes. "
                           "Note: option is expensive when creating lots of "
                           "tests (default=false)"),
                  cl::cat(TestGenCat));

cl::opt<bool>
    SilentKleeAssume("silent-klee-assume", cl::init(false),
                     cl::desc("Silently terminate paths with an infeasible "
                              "condition given to klee_assume rather than "
                              "emitting an error (default=false)"),
                     cl::cat(TerminationCat));
} // namespace

/// \todo Almost all of the demands in this file should be replaced
/// with terminateState calls.

///

// FIXME: We are more or less committed to requiring an intrinsic
// library these days. We can move some of this stuff there,
// especially things like realloc which have complicated semantics
// w.r.t. forking. Among other things this makes delayed query
// dispatch easier to implement.
static constexpr std::array handlerInfo = {
#define add(name, handler, ret) SpecialFunctionHandler::HandlerInfo{ name, \
                                  &SpecialFunctionHandler::handler, \
                                  false, ret, false }
#define addDNR(name, handler) SpecialFunctionHandler::HandlerInfo{ name, \
                                &SpecialFunctionHandler::handler, \
                                true, false, false }
  addDNR("__assert_rtn", handleAssertFail),
  addDNR("__assert_fail", handleAssertFail),
  addDNR("__assert", handleAssertFail),
  addDNR("_assert", handleAssert),
  addDNR("abort", handleAbort),
  addDNR("_exit", handleExit),
  SpecialFunctionHandler::HandlerInfo{ "exit", &SpecialFunctionHandler::handleExit, true, false, true },
  addDNR("klee_abort", handleAbort),
  addDNR("klee_silent_exit", handleSilentExit),
  addDNR("klee_report_error", handleReportError),
  add("aligned_alloc", handleMemalign, true),
  add("calloc", handleCalloc, true),
  add("free", handleFree, false),
  add("klee_assume", handleAssume, false),
  add("klee_check_memory_access", handleCheckMemoryAccess, false),
  add("klee_get_valuef", handleGetValue, true),
  add("klee_get_valued", handleGetValue, true),
  add("klee_get_valuel", handleGetValue, true),
  add("klee_get_valuell", handleGetValue, true),
  add("klee_get_value_i32", handleGetValue, true),
  add("klee_get_value_i64", handleGetValue, true),
  add("klee_define_fixed_object", handleDefineFixedObject, false),
  add("klee_get_obj_size", handleGetObjSize, true),
  add("klee_get_errno", handleGetErrno, true),
#ifndef __APPLE__
  add("__errno_location", handleErrnoLocation, true),
#else
  add("__error", handleErrnoLocation, true),
#endif
  add("klee_is_symbolic", handleIsSymbolic, true),
  add("klee_make_symbolic", handleMakeSymbolic, false),
  add("klee_mark_global", handleMarkGlobal, false),
  add("klee_open_merge", handleOpenMerge, false),
  add("klee_close_merge", handleCloseMerge, false),
  add("klee_prefer_cex", handlePreferCex, false),
  add("klee_posix_prefer_cex", handlePosixPreferCex, false),
  add("klee_print_expr", handlePrintExpr, false),
  add("klee_print_range", handlePrintRange, false),
  add("klee_set_forking", handleSetForking, false),
  add("klee_stack_trace", handleStackTrace, false),
  add("klee_warning", handleWarning, false),
  add("klee_warning_once", handleWarningOnce, false),
  add("malloc", handleMalloc, true),
  add("memalign", handleMemalign, true),
  add("realloc", handleRealloc, true),
  add("cudaMallocManaged", handleCudaMalloc, true), 
  add("cudaMalloc", handleCudaMalloc, true), 
  add("cudaMemcpy", handleCudaMemcpy, true), 
  add("cudaFree", handleCudaFree, true), 
  add("cudaLaunchKernel", handleCudaLaunchKernel, true),
  add("__cudaPopCallConfiguration", handleCudaPopCallConfiguration, true),
  add("__cudaPushCallConfiguration", handleCudaPushCallConfiguration, true),
  add("cudaDeviceSynchronize", handleCudaDeviceSynchronize, true),
  add("cudaDeviceGetAttribute", handleCudaDeviceGetAttribute, true),
  add("cudaFuncSetAttribute", handleCudaFuncSetAttribute, true),
  add("cudaMemcpyAsync", handleCudaMemcpyAsync, true),

#ifdef SUPPORT_KLEE_EH_CXX
  add("_klee_eh_Unwind_RaiseException_impl", handleEhUnwindRaiseExceptionImpl, false),
  add("klee_eh_typeid_for", handleEhTypeid, true),
#endif

  add("_ZdaPv", handleDeleteArray, false),
  add("_ZdlPv", handleDelete, false),

  add("_Znaj", handleNewArray, true),
  add("_Znwj", handleNew, true),

  // FIXME-64: This is wrong for 64-bit long...

  add("_Znam", handleNewArray, true),
  add("_Znwm", handleNew, true),

#undef addDNR
#undef add
};

SpecialFunctionHandler::SpecialFunctionHandler(Executor &_executor) 
  : executor(_executor) {}

void SpecialFunctionHandler::prepare(
    std::vector<const char *> &preservedFunctions) {
  for (auto &hi : handlerInfo) {
    Function *f = executor.kmodule->module->getFunction(hi.name);

    // No need to create if the function doesn't exist, since it cannot
    // be called in that case.
    if (f && (!hi.doNotOverride || f->isDeclaration())) {
      preservedFunctions.push_back(hi.name);
      // Make sure NoReturn attribute is set, for optimization and
      // coverage counting.
      if (hi.doesNotReturn)
        f->addFnAttr(Attribute::NoReturn);

      // Change to a declaration since we handle internally (simplifies
      // module and allows deleting dead code).
      if (!f->isDeclaration())
        f->deleteBody();
    }
  }
}

void SpecialFunctionHandler::bind() {
  for (auto &hi : handlerInfo) {
    Function *f = executor.kmodule->module->getFunction(hi.name);

    if (f && (!hi.doNotOverride || f->isDeclaration()))
      handlers[f] = std::make_pair(hi.handler, hi.hasReturnValue);
  }
}


bool SpecialFunctionHandler::handle(ExecutionState &state, 
                                    Function *f,
                                    KInstruction *target,
                                    std::vector< ref<Expr> > &arguments) {
  handlers_ty::iterator it = handlers.find(f);
  if (it != handlers.end()) {    
    Handler h = it->second.first;
    bool hasReturnValue = it->second.second;
     // FIXME: Check this... add test?
    if (!hasReturnValue && !target->inst->use_empty()) {
      executor.terminateStateOnExecError(state, 
                                         "expected return value from void special function");
    } else {
      (this->*h)(state, target, arguments);
    }
    return true;
  } else {
    return false;
  }
}

/****/

// reads a concrete string from memory
std::string 
SpecialFunctionHandler::readStringAtAddress(ExecutionState &state, 
                                            ref<Expr> addressExpr) {
  ObjectPair op;
  addressExpr = executor.toUnique(state, addressExpr);
  if (!isa<ConstantExpr>(addressExpr)) {
    executor.terminateStateOnUserError(
        state, "Symbolic string pointer passed to one of the klee_ functions");
    return "";
  }
  ref<ConstantExpr> address = cast<ConstantExpr>(addressExpr);
  if (!state.addressSpace.resolveOne(address, op)) {
    executor.terminateStateOnUserError(
        state, "Invalid string pointer passed to one of the klee_ functions");
    return "";
  }
  const MemoryObject *mo = op.first;
  const ObjectState *os = op.second;

  auto relativeOffset = mo->getOffsetExpr(address);
  // the relativeOffset must be concrete as the address is concrete
  size_t offset = cast<ConstantExpr>(relativeOffset)->getZExtValue();

  std::ostringstream buf;
  char c = 0;
  for (size_t i = offset; i < mo->size; ++i) {
    ref<Expr> cur = os->read8(i);
    cur = executor.toUnique(state, cur);
    assert(isa<ConstantExpr>(cur) && 
           "hit symbolic char while reading concrete string");
    c = cast<ConstantExpr>(cur)->getZExtValue(8);
    if (c == '\0') {
      // we read the whole string
      break;
    }

    buf << c;
  }

  if (c != '\0') {
      klee_warning_once(0, "String not terminated by \\0 passed to "
                           "one of the klee_ functions");
  }

  return buf.str();
}

/****/

void SpecialFunctionHandler::handleAbort(ExecutionState &state,
                                         KInstruction *target,
                                         std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 0 && "invalid number of arguments to abort");
  executor.terminateStateOnProgramError(state, "abort failure",
                                        StateTerminationType::Abort);
}

void SpecialFunctionHandler::handleExit(ExecutionState &state,
                                        KInstruction *target,
                                        std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 1 && "invalid number of arguments to exit");
  executor.terminateStateOnExit(state);
}

void SpecialFunctionHandler::handleSilentExit(
    ExecutionState &state, KInstruction *target,
    std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 1 && "invalid number of arguments to exit");
  executor.terminateStateEarlyUser(state, "");
}

void SpecialFunctionHandler::handleAssert(ExecutionState &state,
                                          KInstruction *target,
                                          std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 3 && "invalid number of arguments to _assert");
  executor.terminateStateOnProgramError(
      state, "ASSERTION FAIL: " + readStringAtAddress(state, arguments[0]),
      StateTerminationType::Assert);
}

void SpecialFunctionHandler::handleAssertFail(
    ExecutionState &state, KInstruction *target,
    std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 4 &&
         "invalid number of arguments to __assert_fail");
  executor.terminateStateOnProgramError(
      state, "ASSERTION FAIL: " + readStringAtAddress(state, arguments[0]),
      StateTerminationType::Assert);
}

void SpecialFunctionHandler::handleReportError(
    ExecutionState &state, KInstruction *target,
    std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 4 &&
         "invalid number of arguments to klee_report_error");

  // arguments[0,1,2,3] are file, line, message, suffix
  executor.terminateStateOnProgramError(
      state, readStringAtAddress(state, arguments[2]),
      StateTerminationType::ReportError, "",
      readStringAtAddress(state, arguments[3]).c_str());
}

void SpecialFunctionHandler::handleOpenMerge(ExecutionState &state,
    KInstruction *target,
    std::vector<ref<Expr> > &arguments) {
  if (!UseMerge) {
    klee_warning_once(0, "klee_open_merge ignored, use '-use-merge'");
    return;
  }

  if (state.inMergeRegion) {
    return;
  }
  state.inMergeRegion = true;

  state.openMergeStack.push_back(
      ref<MergeHandler>(new MergeHandler(&executor, &state)));

  if (DebugLogMerge)
    llvm::errs() << "open merge: " << &state << "\n";
}

void SpecialFunctionHandler::handleCloseMerge(ExecutionState &state,
    KInstruction *target,
    std::vector<ref<Expr> > &arguments) {
  if (!UseMerge) {
    klee_warning_once(0, "klee_close_merge ignored, use '-use-merge'");
    return;
  }
  if (!state.inMergeRegion) {
    return;
  }
  Instruction *i = target->inst;

  if (DebugLogMerge)
    llvm::errs() << "close merge: " << &state << " at [" << *i << "]\n";

  if (state.openMergeStack.empty()) {
    std::ostringstream warning;
    warning << &state << " ran into a close at " << i << " without a preceding open";
    klee_warning("%s", warning.str().c_str());
  } else {
    assert(executor.mergingSearcher->inCloseMerge.find(&state) ==
               executor.mergingSearcher->inCloseMerge.end() &&
           "State cannot run into close_merge while being closed");
    executor.mergingSearcher->inCloseMerge.insert(&state);
    state.openMergeStack.back()->addClosedState(&state, i);
    state.openMergeStack.pop_back();
  }
  state.inMergeRegion = false;
}

void SpecialFunctionHandler::handleNew(ExecutionState &state,
                         KInstruction *target,
                         std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to new");

  executor.executeAlloc(state, arguments[0], false, target);
}

void SpecialFunctionHandler::handleDelete(ExecutionState &state,
                            KInstruction *target,
                            std::vector<ref<Expr> > &arguments) {
  // FIXME: Should check proper pairing with allocation type (malloc/free,
  // new/delete, new[]/delete[]).

  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to delete");
  executor.executeFree(state, arguments[0]);
}

void SpecialFunctionHandler::handleNewArray(ExecutionState &state,
                              KInstruction *target,
                              std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to new[]");
  executor.executeAlloc(state, arguments[0], false, target);
}

void SpecialFunctionHandler::handleDeleteArray(ExecutionState &state,
                                 KInstruction *target,
                                 std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to delete[]");
  executor.executeFree(state, arguments[0]);
}

void SpecialFunctionHandler::handleMalloc(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 && "invalid number of arguments to malloc");
  executor.executeAlloc(state, arguments[0], false, target);
}

void SpecialFunctionHandler::handleMemalign(ExecutionState &state,
                                            KInstruction *target,
                                            std::vector<ref<Expr>> &arguments) {
  if (arguments.size() != 2) {
    executor.terminateStateOnUserError(state,
      "Incorrect number of arguments to memalign(size_t alignment, size_t size)");
    return;
  }

  std::pair<ref<Expr>, ref<Expr>> alignmentRangeExpr =
      executor.solver->getRange(state.constraints, arguments[0],
                                state.queryMetaData);
  ref<Expr> alignmentExpr = alignmentRangeExpr.first;
  auto alignmentConstExpr = dyn_cast<ConstantExpr>(alignmentExpr);

  if (!alignmentConstExpr) {
    executor.terminateStateOnUserError(state, "Could not determine size of symbolic alignment");
    return;
  }

  uint64_t alignment = alignmentConstExpr->getZExtValue();

  // Warn, if the expression has more than one solution
  if (alignmentRangeExpr.first != alignmentRangeExpr.second) {
    klee_warning_once(
        0, "Symbolic alignment for memalign. Choosing smallest alignment");
  }

  executor.executeAlloc(state, arguments[1], false, target, false, 0,
                        alignment);
}

#ifdef SUPPORT_KLEE_EH_CXX
void SpecialFunctionHandler::handleEhUnwindRaiseExceptionImpl(
    ExecutionState &state, KInstruction *target,
    std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 1 &&
         "invalid number of arguments to _klee_eh_Unwind_RaiseException_impl");

  ref<ConstantExpr> exceptionObject = dyn_cast<ConstantExpr>(arguments[0]);
  if (!exceptionObject.get()) {
    executor.terminateStateOnExecError(state, "Internal error: Symbolic exception pointer");
    return;
  }

  if (isa_and_nonnull<SearchPhaseUnwindingInformation>(
          state.unwindingInformation.get())) {
    executor.terminateStateOnExecError(
        state,
        "Internal error: Unwinding restarted during an ongoing search phase");
    return;
  }

  state.unwindingInformation =
      std::make_unique<SearchPhaseUnwindingInformation>(exceptionObject,
                                                        state.stack.size() - 1);

  executor.unwindToNextLandingpad(state);
}

void SpecialFunctionHandler::handleEhTypeid(ExecutionState &state,
                                            KInstruction *target,
                                            std::vector<ref<Expr>> &arguments) {
  assert(arguments.size() == 1 &&
         "invalid number of arguments to klee_eh_typeid_for");

  executor.bindLocal(target, state, executor.getEhTypeidFor(arguments[0]));
}
#endif // SUPPORT_KLEE_EH_CXX

void SpecialFunctionHandler::handleAssume(ExecutionState &state,
                            KInstruction *target,
                            std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to klee_assume");
  
  ref<Expr> e = arguments[0];
  
  if (e->getWidth() != Expr::Bool)
    e = NeExpr::create(e, ConstantExpr::create(0, e->getWidth()));
  
  bool res;
  bool success __attribute__((unused)) = executor.solver->mustBeFalse(
      state.constraints, e, res, state.queryMetaData);
  assert(success && "FIXME: Unhandled solver failure");
  if (res) {
    executor.terminateStateOnUserError(
        state, "invalid klee_assume call (provably false)", !SilentKleeAssume);
  } else {
    executor.addConstraint(state, e);
  }
}

void SpecialFunctionHandler::handleIsSymbolic(ExecutionState &state,
                                KInstruction *target,
                                std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to klee_is_symbolic");

  executor.bindLocal(target, state, 
                     ConstantExpr::create(!isa<ConstantExpr>(arguments[0]),
                                          Expr::Int32));
}

void SpecialFunctionHandler::handlePreferCex(ExecutionState &state,
                                             KInstruction *target,
                                             std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_prefex_cex");

  ref<Expr> cond = arguments[1];
  if (cond->getWidth() != Expr::Bool)
    cond = NeExpr::create(cond, ConstantExpr::alloc(0, cond->getWidth()));

  state.addCexPreference(cond);
}

void SpecialFunctionHandler::handlePosixPreferCex(ExecutionState &state,
                                             KInstruction *target,
                                             std::vector<ref<Expr> > &arguments) {
  if (ReadablePosix)
    return handlePreferCex(state, target, arguments);
}

void SpecialFunctionHandler::handlePrintExpr(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_print_expr");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  llvm::errs() << msg_str << ":" << arguments[1] << "\n";
}

void SpecialFunctionHandler::handleSetForking(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_set_forking");
  ref<Expr> value = executor.toUnique(state, arguments[0]);
  
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(value)) {
    state.forkDisabled = CE->isZero();
  } else {
    executor.terminateStateOnUserError(state, "klee_set_forking requires a constant arg");
  }
}

void SpecialFunctionHandler::handleStackTrace(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<ref<Expr> > &arguments) {
  state.dumpStack(outs());
}

void SpecialFunctionHandler::handleWarning(ExecutionState &state,
                                           KInstruction *target,
                                           std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 && "invalid number of arguments to klee_warning");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  klee_warning("%s: %s", state.stack.back().kf->function->getName().data(), 
               msg_str.c_str());
}

void SpecialFunctionHandler::handleWarningOnce(ExecutionState &state,
                                               KInstruction *target,
                                               std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_warning_once");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  klee_warning_once(0, "%s: %s", state.stack.back().kf->function->getName().data(),
                    msg_str.c_str());
}

void SpecialFunctionHandler::handlePrintRange(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_print_range");

  std::string msg_str = readStringAtAddress(state, arguments[0]);
  llvm::errs() << msg_str << ":" << arguments[1];
  if (!isa<ConstantExpr>(arguments[1])) {
    // FIXME: Pull into a unique value method?
    ref<ConstantExpr> value;
    bool success __attribute__((unused)) = executor.solver->getValue(
        state.constraints, arguments[1], value, state.queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
    bool res;
    success = executor.solver->mustBeTrue(state.constraints,
                                          EqExpr::create(arguments[1], value),
                                          res, state.queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
    if (res) {
      llvm::errs() << " == " << value;
    } else { 
      llvm::errs() << " ~= " << value;
      std::pair<ref<Expr>, ref<Expr>> res = executor.solver->getRange(
          state.constraints, arguments[1], state.queryMetaData);
      llvm::errs() << " (in [" << res.first << ", " << res.second <<"])";
    }
  }
  llvm::errs() << "\n";
}

void SpecialFunctionHandler::handleGetObjSize(ExecutionState &state,
                                  KInstruction *target,
                                  std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_get_obj_size");
  Executor::ExactResolutionList rl;
  executor.resolveExact(state, arguments[0], rl, "klee_get_obj_size");
  for (Executor::ExactResolutionList::iterator it = rl.begin(), 
         ie = rl.end(); it != ie; ++it) {
    executor.bindLocal(
        target, *it->second,
        ConstantExpr::create(it->first.first->size,
                             executor.kmodule->targetData->getTypeSizeInBits(
                                 target->inst->getType())));
  }
}

void SpecialFunctionHandler::handleGetErrno(ExecutionState &state,
                                            KInstruction *target,
                                            std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==0 &&
         "invalid number of arguments to klee_get_errno");
#ifndef WINDOWS
  int *errno_addr = executor.getErrnoLocation(state);
#else
  int *errno_addr = nullptr;
#endif

  // Retrieve the memory object of the errno variable
  ObjectPair result;
  bool resolved = state.addressSpace.resolveOne(
      ConstantExpr::create((uint64_t)errno_addr, Expr::Int64), result);
  if (!resolved)
    executor.terminateStateOnUserError(state, "Could not resolve address for errno");
  executor.bindLocal(target, state, result.second->read(0, Expr::Int32));
}

void SpecialFunctionHandler::handleErrnoLocation(
    ExecutionState &state, KInstruction *target,
    std::vector<ref<Expr> > &arguments) {
  // Returns the address of the errno variable
  assert(arguments.size() == 0 &&
         "invalid number of arguments to __errno_location/__error");

#ifndef WINDOWS
  int *errno_addr = executor.getErrnoLocation(state);
#else
  int *errno_addr = nullptr;
#endif

  executor.bindLocal(
      target, state,
      ConstantExpr::create((uint64_t)errno_addr,
                           executor.kmodule->targetData->getTypeSizeInBits(
                               target->inst->getType())));
}
void SpecialFunctionHandler::handleCalloc(ExecutionState &state,
                            KInstruction *target,
                            std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==2 &&
         "invalid number of arguments to calloc");

  ref<Expr> size = MulExpr::create(arguments[0],
                                   arguments[1]);
  executor.executeAlloc(state, size, false, target, true);
}

void SpecialFunctionHandler::handleRealloc(ExecutionState &state,
                            KInstruction *target,
                            std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==2 &&
         "invalid number of arguments to realloc");
  ref<Expr> address = arguments[0];
  ref<Expr> size = arguments[1];

  Executor::StatePair zeroSize =
      executor.fork(state, Expr::createIsZero(size), true, BranchType::Realloc);

  if (zeroSize.first) { // size == 0
    executor.executeFree(*zeroSize.first, address, target);   
  }
  if (zeroSize.second) { // size != 0
    Executor::StatePair zeroPointer =
        executor.fork(*zeroSize.second, Expr::createIsZero(address), true,
                      BranchType::Realloc);

    if (zeroPointer.first) { // address == 0
      executor.executeAlloc(*zeroPointer.first, size, false, target);
    } 
    if (zeroPointer.second) { // address != 0
      Executor::ExactResolutionList rl;
      executor.resolveExact(*zeroPointer.second, address, rl, "realloc");
      
      for (Executor::ExactResolutionList::iterator it = rl.begin(), 
             ie = rl.end(); it != ie; ++it) {
        executor.executeAlloc(*it->second, size, false, target, false, 
                              it->first.second);
      }
    }
  }
}

void SpecialFunctionHandler::handleFree(ExecutionState &state,
                          KInstruction *target,
                          std::vector<ref<Expr> > &arguments) {
  // XXX should type check args
  assert(arguments.size()==1 &&
         "invalid number of arguments to free");
  executor.executeFree(state, arguments[0]);
}

void SpecialFunctionHandler::handleCheckMemoryAccess(ExecutionState &state,
                                                     KInstruction *target,
                                                     std::vector<ref<Expr> > 
                                                       &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_check_memory_access");

  ref<Expr> address = executor.toUnique(state, arguments[0]);
  ref<Expr> size = executor.toUnique(state, arguments[1]);
  if (!isa<ConstantExpr>(address) || !isa<ConstantExpr>(size)) {
    executor.terminateStateOnUserError(state, "check_memory_access requires constant args");
  } else {
    ObjectPair op;

    if (!state.addressSpace.resolveOne(cast<ConstantExpr>(address), op)) {
      executor.terminateStateOnProgramError(
          state, "check_memory_access: memory error", StateTerminationType::Ptr,
          executor.getAddressInfo(state, address));
    } else {
      ref<Expr> chk = 
        op.first->getBoundsCheckPointer(address, 
                                        cast<ConstantExpr>(size)->getZExtValue());
      if (!chk->isTrue()) {
        executor.terminateStateOnProgramError(
            state, "check_memory_access: memory error",
            StateTerminationType::Ptr, executor.getAddressInfo(state, address));
      }
    }
  }
}

void SpecialFunctionHandler::handleGetValue(ExecutionState &state,
                                            KInstruction *target,
                                            std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_get_value");

  executor.executeGetValue(state, arguments[0], target);
}

void SpecialFunctionHandler::handleDefineFixedObject(ExecutionState &state,
                                                     KInstruction *target,
                                                     std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==2 &&
         "invalid number of arguments to klee_define_fixed_object");
  assert(isa<ConstantExpr>(arguments[0]) &&
         "expect constant address argument to klee_define_fixed_object");
  assert(isa<ConstantExpr>(arguments[1]) &&
         "expect constant size argument to klee_define_fixed_object");
  
  uint64_t address = cast<ConstantExpr>(arguments[0])->getZExtValue();
  uint64_t size = cast<ConstantExpr>(arguments[1])->getZExtValue();
  MemoryObject *mo = executor.memory->allocateFixed(address, size, state.prevPC->inst);
  executor.bindObjectInState(state, mo, false);
  mo->isUserSpecified = true; // XXX hack;
}

void SpecialFunctionHandler::handleMakeSymbolic(ExecutionState &state,
                                                KInstruction *target,
                                                std::vector<ref<Expr> > &arguments) {
  std::string name;

  if (arguments.size() != 3) {
    executor.terminateStateOnUserError(state,
        "Incorrect number of arguments to klee_make_symbolic(void*, size_t, char*)");
    return;
  }

  name = arguments[2]->isZero() ? "" : readStringAtAddress(state, arguments[2]);

  if (name.length() == 0) {
    name = "unnamed";
    klee_warning("klee_make_symbolic: renamed empty name to \"unnamed\"");
  }

  Executor::ExactResolutionList rl;
  executor.resolveExact(state, arguments[0], rl, "make_symbolic");
  
  for (Executor::ExactResolutionList::iterator it = rl.begin(), 
         ie = rl.end(); it != ie; ++it) {
    const MemoryObject *mo = it->first.first;
    mo->setName(name);
    
    const ObjectState *old = it->first.second;
    ExecutionState *s = it->second;
    
    if (old->readOnly) {
      executor.terminateStateOnUserError(*s, "cannot make readonly object symbolic");
      return;
    } 

    // FIXME: Type coercion should be done consistently somewhere.
    bool res;
    bool success __attribute__((unused)) = executor.solver->mustBeTrue(
        s->constraints,
        EqExpr::create(
            ZExtExpr::create(arguments[1], Context::get().getPointerWidth()),
            mo->getSizeExpr()),
        res, s->queryMetaData);
    assert(success && "FIXME: Unhandled solver failure");
    
    if (res) {
      executor.executeMakeSymbolic(*s, mo, name);
    } else {      
      executor.terminateStateOnUserError(*s, "Wrong size given to klee_make_symbolic");
    }
  }
}

void SpecialFunctionHandler::handleMarkGlobal(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<ref<Expr> > &arguments) {
  assert(arguments.size()==1 &&
         "invalid number of arguments to klee_mark_global");  

  Executor::ExactResolutionList rl;
  executor.resolveExact(state, arguments[0], rl, "mark_global");
  
  for (Executor::ExactResolutionList::iterator it = rl.begin(), 
         ie = rl.end(); it != ie; ++it) {
    const MemoryObject *mo = it->first.first;
    assert(!mo->isLocal);
    mo->isGlobal = true;
  }
}

/// Define the handler for cudaMalloc
void SpecialFunctionHandler::handleCudaMalloc(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<ref<Expr>> &arguments) {

    executor.executeAlloc(state, arguments[1], false, target, false, nullptr, 0, arguments[0]);
}

void SpecialFunctionHandler::handleCudaMemcpy(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<ref<Expr>> &arguments) {
    assert(arguments.size() == 4 && "Invalid number of arguments to cudaMemcpy"); 

    ref<Expr> dstPtrExpr = arguments[0];  // Destination pointer (symbolic)
    ref<Expr> srcPtrExpr = arguments[1];  // Source pointer (symbolic)
    ref<Expr> sizeExpr = arguments[2];    // Size of memory to copy (symbolic or concrete)
    ref<Expr> memcpyKindExpr = arguments[3];  // Direction of the copy (concrete, enum type)

    // Check the memcpy direction - it should be a constant expression
    ConstantExpr *memcpyKind = dyn_cast<ConstantExpr>(memcpyKindExpr);
    if (!memcpyKind) {
        // If the copy kind isn't constant, we can't handle it
        executor.bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
        return;
    }

    uint64_t memcpyKindValue = memcpyKind->getZExtValue();

    // Depending on the type of copy, decide how to handle it
    switch (memcpyKindValue) {
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
            executor.executeCudaMemcpy(state, dstPtrExpr, srcPtrExpr, sizeExpr, target);
            executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
            break;
        default:
            // If the memory transfer type is invalid, return an error
            executor.bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
    }
}

void SpecialFunctionHandler::handleCudaFree(ExecutionState &state,
                                              KInstruction *target,
                                              std::vector<ref<Expr>> &arguments) {

  assert(arguments.size()==1 &&
         "invalid number of arguments to free");
  executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
}                                              

void SpecialFunctionHandler::handleCudaLaunchKernel(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {
  klee_warning("handle cudaLaunchKernel");

  ref<Expr> functionPointerExpr = arguments[0];
  ref<ConstantExpr> concreteFuncPtr = dyn_cast<ConstantExpr>(functionPointerExpr);
  std::string keyword = "__device_stub__";
  std::string funcName;
  if (concreteFuncPtr) {
    uint64_t addr = concreteFuncPtr->getZExtValue();
    auto it = executor.legalFunctions.find(addr);
    if (it != executor.legalFunctions.end()) {
      Function *f = it->second;
      funcName = f->getName().str();
    } else {
      executor.terminateStateOnProgramError(state, "cudaLaunchKernel: no function found",
                                         StateTerminationType::ReportError);
      return;     
    }
  } else {
    executor.terminateStateOnProgramError(state, "function pointer not constant",
                                         StateTerminationType::ReportError);
    return;     
  }
  if (funcName.empty()) {
    executor.terminateStateOnProgramError(state, "cudaLaunchKernel function name empty",
                                         StateTerminationType::ReportError);
    return;     
  }
  if (funcName.find(keyword) == std::string::npos) {
    executor.terminateStateOnProgramError(state, "cudaLaunchKernel function name not contain __device__stub",
                                         StateTerminationType::ReportError);
    return;     
  }

  ref<Expr> argsPointerExpr = arguments[5];
  bool success;
  ObjectPair op0;
  if (!state.addressSpace.resolveOne(state, executor.solver.get(), argsPointerExpr, op0, success) || !success) {
    executor.terminateStateOnProgramError(state, "cudaLaunchKernel: do not find the parameter array object",
                                         StateTerminationType::ReportError);
    return;                                        
  }
  
  const MemoryObject *argMo = op0.first;
  unsigned numArgs = argMo->size/8;
  std::vector< ref<Expr> > passedArguments;
  passedArguments.reserve(numArgs);

  std::string realFunctionName = funcName.substr(funcName.find(keyword) + keyword.length());
  std::string kName;
  Function* kernelf;
  llvm::FunctionType* funcType;
  for (const auto &kernelName : executor.kernelNames) {
    if (kernelName.size() < realFunctionName.size()) continue;
    if (std::equal(realFunctionName.rbegin(), realFunctionName.rend(), kernelName.rbegin())) {
      kName = kernelName;
      kernelf = executor.getFunctionByName(kName);
      funcType = kernelf->getFunctionType();
      if (funcType->getNumParams()!=numArgs) {
        kName = "";
        kernelf = nullptr;
        funcType = nullptr;
      } else {
        break;
      }
    }
  }
  if (kName.empty()) {
    executor.terminateStateOnProgramError(state, "cudaLaunchKernel: not found corresponding kernel function name",
                                         StateTerminationType::ReportError);
    return;     
  }

  if (!kernelf) {
    executor.terminateStateOnProgramError(state, "cudaLaunchKernel: target function not found",
                                         StateTerminationType::ReportError);
    return;
  }

  llvm::AttributeList attrList = kernelf->getAttributes();
  const ObjectState *argOS = op0.second;
  for (unsigned i = 0; i < numArgs; ++i) {
    llvm::Type* argType = funcType->getParamType(i);
    ref<Expr> argExpr = argOS->read(i*8, Expr::Int64);

    if (!attrList.hasParamAttr(i, llvm::Attribute::ByVal) && !argExpr.isNull()) {
      ObjectPair op1;
      auto base_it = state.base_addrs.find(argExpr);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, executor.solver.get(), base_it->second, op1,
                                          success) ||
            !success) {
          executor.terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addrs", StateTerminationType::ReportError);
          return;
        }
      } else {
        if (!state.addressSpace.resolveOne(state, executor.solver.get(), argExpr, op1, success) || !success) {
          executor.terminateStateOnProgramError(state, "cudaLaunchKernel: do not find argument object", StateTerminationType::ReportError);
          return;
        }
      }
      argExpr = op1.second->read(0, op1.first->size*8);
    }
    passedArguments.push_back(argExpr);

    if (argType->isPointerTy() && !argExpr.isNull()) {
      ObjectPair op1;
      auto base_it = state.base_addrs.find(argExpr);
      if (base_it != state.base_addrs.end()) {
        if (!state.addressSpace.resolveOne(state, executor.solver.get(), base_it->second, op1,
                                          success) ||
            !success) {
          executor.terminateStateOnProgramError(state, "Failed to resolve concrete address from the base_addrs", StateTerminationType::ReportError);
          return;
        }
      } else {
        if (!state.addressSpace.resolveOne(state, executor.solver.get(), argExpr, op1, success) || !success) {
          continue;
        }
      }
      
      llvm::StructType *structType = dyn_cast<llvm::StructType>(argType->getPointerElementType());
      if (structType && structType->getName().find("detail::TensorInfo")!=std::string::npos) {
        ref<Expr> dataAddress = op1.second->read(0, 64);
        ObjectPair op2;
        if (!state.addressSpace.resolveOne(state, executor.solver.get(), dataAddress, op2, success) || !success) {
          executor.terminateStateOnProgramError(state, "cudaLaunchKernel: do not find tensor object", StateTerminationType::ReportError);
          return;
        }
        op2.first->setMemType(MemoryObject::MemType::GLOBAL);
      }
      op1.first->setMemType(MemoryObject::MemType::GLOBAL);
    }
  }
  
  executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
  executor.executeCall(state, target, kernelf, passedArguments);
}

void SpecialFunctionHandler::handleCudaPopCallConfiguration(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {
    klee_warning("ignore __cudaPopCallConfiguration");

    executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
}

void SpecialFunctionHandler::handleCudaPushCallConfiguration(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {

    ref<Expr> gridExpr = arguments[0];
    auto gridxExpr = klee::ExtractExpr::create(gridExpr, 0, 32);  // Lower 32 bits
    auto gridyExpr = klee::ExtractExpr::create(gridExpr, 32, 32); // Upper 32 bits
    ref<Expr> gridzExpr = arguments[1];

    ref<Expr> blockExpr = arguments[2];
    auto blockxExpr = klee::ExtractExpr::create(blockExpr, 0, 32);  // Lower 32 bits
    auto blockyExpr = klee::ExtractExpr::create(blockExpr, 32, 32); // Upper 32 bits
    ref<Expr> blockzExpr = arguments[3];

    ref<Expr> sharedMemSizeExpr = arguments[4];

    executor.initializeCudaKernelConfig(state, gridxExpr, gridyExpr, gridzExpr, blockxExpr, blockyExpr, blockzExpr, sharedMemSizeExpr);
    executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
}

void SpecialFunctionHandler::handleCudaDeviceSynchronize(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {
    klee_warning("ignore cudaDeviceSynchronize");

    executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
}

void SpecialFunctionHandler::handleCudaDeviceGetAttribute(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {
    ref<Expr> address = arguments[0];
    ref<Expr> kindExpr = arguments[1];
    ref<Expr> value;
    if (auto kindCE = dyn_cast<ConstantExpr>(kindExpr)) {
      int kind = kindCE->getZExtValue();
      if (kind == 1) {
        value = ConstantExpr::create(1024, Expr::Int32);
      } else if (kind == 16) {
        value = ConstantExpr::create(80, Expr::Int32);
      } else if (kind == 97) {
        value = ConstantExpr::create(228*1024, Expr::Int32);
      } else if (kind == 75) {
        value = ConstantExpr::create(9, Expr::Int32);
      } else if (kind == 76) {
        value = ConstantExpr::create(0, Expr::Int32);
      } else {
        std::string symName = "device_attr_" + llvm::utostr(kind);
        MemoryObject *mo =
          executor.memory->allocate(4, /*isLocal=*/false, /*isGlobal=*/false,
                          &state, /*allocSite=*/state.prevPC->inst,
                          /*alignment=*/8);          
        value = executor.symbolizeIndex(state, symName, mo, false);
      }
    } else {
      executor.terminateStateOnProgramError(state, "cuda device attribute not constant",
                                         StateTerminationType::ReportError);
      return;     
    }

    if (value) {
      executor.executeMemoryOperation(state, true, address, value, target);
    }
    executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
}

void SpecialFunctionHandler::handleCudaFuncSetAttribute(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {
    klee_warning("ignore cudaFuncSetAttribute");

    executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
}

void SpecialFunctionHandler::handleCudaMemcpyAsync(ExecutionState &state,
                                                    KInstruction *target,
                                                    std::vector<ref<Expr>> &arguments) {
    ref<Expr> dstPtrExpr = arguments[0];  
    ref<Expr> srcPtrExpr = arguments[1];  
    ref<Expr> sizeExpr = arguments[2];    
    ref<Expr> memcpyKindExpr = arguments[3];  

    ref<ConstantExpr> memcpyKind = dyn_cast<ConstantExpr>(memcpyKindExpr);
    if (!memcpyKind) {
        // If the copy kind isn't constant, we can't handle it
        executor.bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
        return;
    }

    uint64_t memcpyKindValue = memcpyKind->getZExtValue();

    switch (memcpyKindValue) {
      case cudaMemcpyHostToHost:
      case cudaMemcpyDeviceToHost:
      case cudaMemcpyDeviceToDevice:
      executor.executeCudaMemcpy(state, dstPtrExpr, srcPtrExpr, sizeExpr, target);
        executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
        break;
      default:
        // If the memory transfer type is invalid, return an error
        executor.bindLocal(target, state, ConstantExpr::create(1, Expr::Int32));
  }
}
