#include <iostream>
#include <vector>
#include <thread>
#include <map>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Attributes.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <filesystem>
#include <cxxabi.h>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <future>
#include <cstdlib>
#include <chrono>

using namespace llvm;
namespace fs = std::filesystem;

cl::opt<std::string>
  InputFile(cl::desc("<input file>"), cl::Positional, cl::init("-"));

cl::list<std::string>
  InputArgv(cl::ConsumeAfter,
            cl::desc("<program arguments>..."));

cl::opt<std::string>
  OutputDir("cuklee-out-dir",
            cl::desc("Directory in which to write results"),
            cl::init(""));

cl::opt<unsigned>
  TimeOutSec("timeout",
            cl::desc("timeout in seconds"),
            cl::init(0));

cl::opt<bool>
  AllowReRun("re-run",
                 cl::desc("allow re-run for same index (default=false)."),
		 cl::init(false));

cl::opt<bool>
  SymLoop("sym-loop",
                 cl::desc("symbolize loop index (default=true)."),
		 cl::init(true));

cl::opt<bool>
  UseMerge("use-merge",
                 cl::desc("Enable support for path merging via klee_open_merge and klee_close_merge (default=true)."),
		 cl::init(true));

cl::opt<unsigned>
  MaxEntry("max-entry",
            cl::desc("maximum number of entries"),
            cl::init(0));


std::map<std::string, std::set<std::string>> buildCallGraph(const llvm::Module &module) {
    // Map from callee function to set of callers
    std::map<std::string, std::set<std::string>> callGraph;

    // Build the call graph by iterating through the module
    for (const auto &func : module) {
        if (func.isDeclaration() || func.empty()) continue;

        const std::string callerName = func.getName().str();
        // To store the memory addresses and their corresponding function names
        std::map<const llvm::Value*, std::set<std::string>> addressToFunction;

        for (const auto &bb : func) {
            for (const auto &inst : bb) {
                const llvm::Function *calledFunc = nullptr;

                if (const auto *callInst = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                    calledFunc = callInst->getCalledFunction();
                } else if (const auto *invokeInst = llvm::dyn_cast<llvm::InvokeInst>(&inst)) {
                    calledFunc = invokeInst->getCalledFunction();
                }

                if (calledFunc) {
                    callGraph[calledFunc->getName().str()].insert(callerName);
                }

                if (const auto *storeInst = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
                    const llvm::Value *storedValue = storeInst->getValueOperand();
                    const llvm::Value *storeAddress = storeInst->getPointerOperand();
                    if (const auto *funcValue = llvm::dyn_cast<llvm::Function>(storedValue)) {
                        addressToFunction[storeAddress].insert(funcValue->getName().str());
                    }
                }

                if (const auto *loadInst = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
                    const llvm::Value *loadAddress = loadInst->getPointerOperand();

                    if (addressToFunction.count(loadAddress)) {
                        for (auto& calleeName: addressToFunction[loadAddress])
                            callGraph[calleeName].insert(callerName);
                    }
                }

                if (const auto *selectInst = llvm::dyn_cast<llvm::SelectInst>(&inst)) {
                    const llvm::Value *trueValue = selectInst->getTrueValue();
                    const llvm::Value *falseValue = selectInst->getFalseValue();

                    if (const auto *trueFunc = llvm::dyn_cast<llvm::Function>(trueValue)) {
                        callGraph[trueFunc->getName().str()].insert(callerName);
                    }
                    if (const auto *falseFunc = llvm::dyn_cast<llvm::Function>(falseValue)) {
                        callGraph[falseFunc->getName().str()].insert(callerName);
                    }
                }
            }
        }
    }
    return callGraph;
}

std::set<std::string> findAllHighestLevelCallers(std::map<std::string, std::set<std::string>> &callGraph, const std::string &functionName) {
    // Traverse the call graph to find all root callers of the function
    std::set<std::string> visited;
    std::set<std::string> highestCallers;
    std::vector<std::string> worklist = {functionName};

    while (!worklist.empty()) {
        std::string currentFunction = worklist.back();
        worklist.pop_back();

        // If we've already visited this function, skip it
        if (visited.count(currentFunction)) continue;

        visited.insert(currentFunction);

        auto &callers = callGraph[currentFunction];
        if (callers.empty()) {
            // If no callers exist, this is a root function
            highestCallers.insert(currentFunction);
        } else {
            // Add all callers to the worklist
            worklist.insert(worklist.end(), callers.begin(), callers.end());
        }
    }

    return highestCallers;
}

std::string getKernelStubName(const llvm::Module &module, const std::string &kernelName) {
    std::string keyword = "__device_stub__";
    for (const auto &func : module) {
        const std::string funcName = func.getName().str();
        if (funcName.find(keyword) != std::string::npos) {
            std::string realFunctionName = funcName.substr(funcName.find(keyword) + keyword.length());
            if (kernelName.find(realFunctionName) != std::string::npos) {
                return funcName;
            }
        }
    }
    return "";
}

// A substitution table that holds previously encoded fragments.
using SubstTable = std::vector<std::string>;

// If 'encoded' is already in 'subs', return its substitution code;
// otherwise, add it and return the full encoding.
std::string substituteOrAdd(const std::string &encoded, SubstTable &subs) {
    for (size_t i = 0; i < subs.size(); ++i) {
        if (subs[i] == encoded) {
            // For index 0, substitution is "S_"; otherwise "S<index>_"
            return ("S" + std::to_string(i) + "_");
        }
    }
    subs.push_back(encoded);
    return encoded;
}

std::string substitute(const std::string &encoded, SubstTable &subs) {
    for (size_t i = 0; i < subs.size(); ++i) {
        if (subs[i] == encoded) {
            // For index 0, substitution is "S_"; otherwise "S<index>_"
            return (i == 0 ? "S_" : ("S" + std::to_string(i) + "_"));
        }
    }
    return encoded;
}

// Encode a simple (unqualified) name as <length><name>.
std::string encodeName(const std::string &name) {
    return std::to_string(name.size()) + name;
}

// Encode a qualified name (for function names) using the substitution table.
std::string encodeQualifiedName(const std::string &qualifiedName, SubstTable &subs) {
    std::string temp = qualifiedName; // local copy
    std::vector<std::string> parts;
    size_t pos = 0;
    while ((pos = temp.find("::")) != std::string::npos) {
        size_t delim = temp.find("::");
        parts.push_back(substituteOrAdd(encodeName(temp.substr(0, delim)), subs));
        temp.erase(0, delim + 2);
    }
    parts.push_back(substituteOrAdd(encodeName(temp), subs));
    std::ostringstream oss;
    oss << "N";
    for (const auto &p : parts)
        oss << p;
    oss << "E";
    return oss.str();
}

// Encode a qualified type name for use in type encoding.
// Here we add a special mapping: if the type is "torch::Tensor", we treat it as "at::Tensor".
std::string encodeQualifiedType(const std::string &qualifiedType, SubstTable &subs) {
    // Apply special mapping.
    std::string effective = qualifiedType;
    if (qualifiedType == "torch::Tensor")
        effective = "at::Tensor";

    std::string temp = effective; // local copy
    std::vector<std::string> parts;
    size_t pos = 0;
    while ((pos = temp.find("::")) != std::string::npos) {
        parts.push_back(temp.substr(0, pos));
        temp.erase(0, pos + 2);
    }
    parts.push_back(temp);
    
    std::ostringstream oss;
    oss << "N";
    if (!parts.empty()) {
        // For the first part, check if it has already been encoded.
        std::string firstPart = encodeName(parts[0]);
        bool found = false;
        for (size_t i = 0; i < subs.size(); ++i) {
            if (subs[i] == firstPart) {
                oss << (i == 0 ? "S_" : ("S" + std::to_string(i) + "_"));
                found = true;
                break;
            }
        }
        if (!found) {
            oss << substitute(firstPart, subs);
        }
        // Encode the rest without substitution.
        for (size_t i = 1; i < parts.size(); ++i) {
            oss << encodeName(parts[i]);
        }
    }
    oss << "E";
    return oss.str();
}

// Parse a type string to detect qualifiers ("const" and trailing '&').
// Returns a pair: <isReference, isConst>
std::pair<bool, bool> parseTypeQualifiers(const std::string &t) {
    bool isRef = false, isConst = false;
    std::string s = t;
    while (!s.empty() && std::isspace(s.front()))
        s.erase(s.begin());
    const std::string constPrefix = "const ";
    if (s.compare(0, constPrefix.size(), constPrefix) == 0) {
        isConst = true;
        s = s.substr(constPrefix.size());
        while (!s.empty() && std::isspace(s.front()))
            s.erase(s.begin());
    }
    if (!s.empty() && s.back() == '&') {
        isRef = true;
        s.pop_back();
        while (!s.empty() && std::isspace(s.back()))
            s.pop_back();
    }
    return {isRef, isConst};
}

// Remove "const" and trailing '&' from the type string.
std::string stripQualifiers(const std::string &t) {
    std::string s = t;
    while (!s.empty() && std::isspace(s.front()))
        s.erase(s.begin());
    while (!s.empty() && std::isspace(s.back()))
        s.pop_back();
    const std::string constPrefix = "const ";
    if (s.compare(0, constPrefix.size(), constPrefix) == 0)
        s = s.substr(constPrefix.size());
    if (!s.empty() && s.back() == '&')
        s.pop_back();
    while (!s.empty() && std::isspace(s.front()))
        s.erase(s.begin());
    while (!s.empty() && std::isspace(s.back()))
        s.pop_back();
    return s;
}

// Map a basic type to an encoding.
std::string mapBasicType(const std::string &baseType) {
    if (baseType == "int")
        return "i";
    if (baseType == "long")
        return "l";
    if (baseType == "int64_t")
        return "l"; // using 'l' for int64_t as expected.
    if (baseType == "long long")
        return "x";
    if (baseType == "void")
        return "v";
    return encodeName(baseType);
}

// Encode a type (including its qualifiers) as a full string,
// then apply substitution. For a qualified base (contains "::"),
// we use encodeQualifiedType to apply our torch->at mapping.
std::string encodeType(const std::string &typeStr, SubstTable &subs) {
    auto quals = parseTypeQualifiers(typeStr);
    bool isRef = quals.first, isConst = quals.second;
    std::string base = stripQualifiers(typeStr);
    std::string typeCode;
    bool needSub = false;
    if (base.find("::") != std::string::npos){
        typeCode = encodeQualifiedType(base, subs);
        needSub = true;
    } else
        typeCode = mapBasicType(base);
    
    std::string full;
    if (isRef) {
        full = "R";
        if (isConst)
            full += "K";
        full += typeCode;
    } else {
        if (isConst)
            full += "K";
        full += typeCode;
    }
    if (needSub)
        return substituteOrAdd(full, subs);
    else
        return full;
}

// Mangle a function name given its qualified name, parameter types, and return type.
// (We ignore the return type for our purposes.)
std::string mangleFunctionName(const std::string &qualifiedName,
                               const std::vector<std::string> &paramTypes,
                               const std::string & /*returnType*/) {
    std::ostringstream oss;
    oss << "_Z"; // Itanium prefix

    SubstTable subs; // Substitution table

    // Encode the nested function name.
    std::vector<std::string> parts;
    {
        std::string temp = qualifiedName;
        size_t pos = 0;
        while ((pos = temp.find("::")) != std::string::npos) {
            parts.push_back(temp.substr(0, pos));
            temp.erase(0, pos + 2);
        }
        parts.push_back(temp);
    }
    if (parts.size() > 1) {
        oss << "N";
        for (const auto &p : parts)
            oss << substituteOrAdd(encodeName(p), subs);
        oss << "E";
    } else {
        oss << encodeName(qualifiedName);
    }

    // Encode parameter types.
    if (paramTypes.empty()) {
        oss << "v";
    } else {
        for (const auto &ptype : paramTypes)
            oss << encodeType(ptype, subs);
    }
    return oss.str();
}

std::string getModifiedFilePath(const std::string &inputFilePath) {
    fs::path inputPath(inputFilePath);
    fs::path outputPath = inputPath.parent_path() / (inputPath.stem().string() + "_modified.bc");
    return outputPath.string();
}

// Function to check if a command exists in the system path
bool commandExists(const std::string &command) {
    std::string checkCmd = "command -v " + command + " > /dev/null 2>&1";
    return (std::system(checkCmd.c_str()) == 0);
}

void runOptCommand(const std::string &inputFilePath, const std::string &outputFilePath) {
    std::filesystem::path currentFilePath = __FILE__;
    std::filesystem::path passLibPath = std::filesystem::absolute(currentFilePath).parent_path().parent_path() / "build/libSignednessPropagationPass.so";
    std::string passLibPathStr = passLibPath.string();

    // Construct the opt command
    std::string optCommand = "opt-13 -load-pass-plugin " + passLibPathStr +
                             " -passes=signedness-prop <" + inputFilePath +
                             " > " + outputFilePath;
    std::cout << "running " << optCommand << "\n";

    // Run the command
    int result = system(optCommand.c_str());
    if (result != 0) {
        std::cerr << "Error running opt with SignednessPropagationPass\n";
        return;
    }

    std::cout << "Pass applied. Modified file: " << outputFilePath << "\n";
}

void runKLEECommand(const std::string &functionName, const std::string &functionType, const std::string &jsonFile, const std::string &filePath, int i, const std::string &outputDir="") {
    // Run KLEE command with callerFunctionName as entry-point
    std::string command = "klee --disable-verify --warnings-only-to-file --single-object-resolution=true --external-calls=over-approx --max-depth=1024 --debug-print-instructions=all:file --entry-point=" + functionName + " --cuda-type=" + functionType + " --func-config=" + jsonFile  + " --jindex=" + llvm::utostr(i);
    if (!outputDir.empty()) {
        command += " --output-dir="+outputDir;
    }
    if (UseMerge) {
        command += " --use-merge";
    }
    if (AllowReRun) {
        command += " --re-run=true";
    }
    if (!SymLoop) {
        command += " --sym-loop=false";
    }
    command += " " + filePath;
    std::cout << "running " << command << "\n";

    if (TimeOutSec != 0) {
        command = "timeout -k 5s " + llvm::utostr(TimeOutSec) + "s " + command;
    }

    int result = system(command.c_str());
    if (result != 0) {
        if (TimeOutSec != 0 && result == 124 * 256) {
            std::cerr << "⚠️ cuKLEE timed out after " << TimeOutSec << " seconds for function " << functionName << std::endl;
        }
        std::cout << std::endl;
        return;
    }
    std::cout << std::endl;
}

void runKLEECommand(const std::string &functionName, const std::string &functionType, const std::string &filePath, const std::string &outputDir="") {
    // Run KLEE command with callerFunctionName as entry-point
    std::string command = "klee --disable-verify --warnings-only-to-file --single-object-resolution=true --external-calls=over-approx --max-depth=1024 --debug-print-instructions=all:file --entry-point=" + functionName + " --cuda-type=" + functionType;
    if (!outputDir.empty()) {
        command += " --output-dir="+outputDir;
    }
    if (UseMerge) {
        command += " --use-merge";
    }
    if (AllowReRun) {
        command += " --re-run=true";
    }
    if (!SymLoop) {
        command += " --sym-loop=false";
    }
    command += " " + filePath;
    std::cout << "running " << command << "\n";

    if (TimeOutSec != 0) {
        command = "timeout -k 5s " + llvm::utostr(TimeOutSec) + "s " + command;
    }

    int result = system(command.c_str());
    if (result != 0) {
        if (TimeOutSec != 0 && result == 124 * 256) {
            std::cerr << "⚠️ cuKLEE timed out after " << TimeOutSec << " seconds for function " << functionName << std::endl;
        }
        std::cout << std::endl;
        return;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "cuKLEE\n");
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <inputFilePath>\n";
        return 1;
    }

    std::string arg1 = InputFile;
    if (arg1.find(".json") != std::string::npos) {
        std::ifstream file(arg1);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file.\n";
            return 1;
        }

        nlohmann::json j;
        file >> j;

        fs::path filePath(arg1);
        fs::path parentPath = filePath.parent_path();
        fs::path projectPath = fs::absolute(fs::path(__FILE__)).parent_path().parent_path().parent_path();

        if(j.is_array()) {
            for (unsigned i = 0; i < j.size(); i++) {
                if (MaxEntry != 0 && i >= MaxEntry) {
                    std::cout << "Reached maximum entry limit (" << MaxEntry << "). Stopping further processing.\n";
                    break;
                }
                auto& c = j[i];
                std::string function = c.value("cuda_function", "");
                std::string inputFilePath = c.value("input_file_path", "");
                if (!inputFilePath.empty() && inputFilePath.front() != '/') {
                    inputFilePath = (projectPath / inputFilePath).string();
                }

                std::string modifiedFilePath = getModifiedFilePath(inputFilePath);
                runOptCommand(inputFilePath, modifiedFilePath);
                LLVMContext context;
                SMDiagnostic error;
                auto module = parseIRFile(modifiedFilePath, error, context);
        
                if (!module) {
                    errs() << "Error reading module\n";
                    continue;
                }

                std::string targetFunctionName;

                for (auto &F : *module) {
                    if (F.isDeclaration()) continue; 

                    std::string mangledName = F.getName().str();

                    // Demangle the name to get the original function name
                    int status = 0;
                    char *demangledName = abi::__cxa_demangle(mangledName.c_str(), nullptr, nullptr, &status);

                    // If demangling failed, use the mangled name
                    std::string demangledNameStr = (status == 0) ? demangledName : mangledName;

                    // Free the demangled name if it was successfully demangled
                    if (status == 0) {
                        free(demangledName);
                    }

                    std::string demangledFuncName = "";
                    size_t pos = demangledNameStr.find('(');
                    if (pos != std::string::npos) {
                        demangledFuncName = demangledNameStr.substr(0, pos);
                    }
                    pos = demangledFuncName.rfind("::");
                    if (pos != std::string::npos) {
                        demangledFuncName = demangledFuncName.substr(pos + 2);
                    }

                    if (demangledFuncName == function) {
                        targetFunctionName = mangledName;
                        break;
                    }
                }
                if (targetFunctionName.empty()) {
                    continue;
                }
                runKLEECommand(targetFunctionName, "caller", arg1, modifiedFilePath, i, OutputDir);
            }
        }
    } else if (arg1.find(".bc") != std::string::npos) {
        std::string modifiedFilePath = getModifiedFilePath(arg1);
        runOptCommand(arg1, modifiedFilePath);

        LLVMContext context;
        SMDiagnostic error;
        auto module = parseIRFile(modifiedFilePath, error, context);
    
        if (!module) {
            errs() << "Error reading module\n";
            return 1;
        }

        std::vector<std::string> kernelFunctionNames;
        std::map<std::string, std::string> functionCallersMap;

        if (const llvm::NamedMDNode *nvvmAnnotations = module->getNamedMetadata("nvvm.annotations")) {

            for (unsigned i = 0; i < nvvmAnnotations->getNumOperands(); ++i) {
                if (const llvm::MDNode *node = nvvmAnnotations->getOperand(i)) {
                    if (node->getNumOperands() > 1) {
                        if (const llvm::MDString *tag = llvm::dyn_cast<llvm::MDString>(node->getOperand(1))) {
                            if (tag->getString() == "kernel") {
                                if (const llvm::Metadata *md = node->getOperand(0).get()) {
                                    if (const llvm::ValueAsMetadata *valueMeta = llvm::dyn_cast<llvm::ValueAsMetadata>(md)) {
                                        if (const llvm::Function *func = llvm::dyn_cast<llvm::Function>(valueMeta->getValue())) {
                                            kernelFunctionNames.emplace_back(func->getName());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::map<std::string, std::set<std::string>> callGraph = buildCallGraph(*module);
        for (const auto &kernelFunctionName : kernelFunctionNames) {
            if (kernelFunctionName.find("EmptyKernel") != std::string::npos) {
                continue;
            }
            std::string stubFuncName = getKernelStubName(*module, kernelFunctionName);
            std::set<std::string> highestCallers = findAllHighestLevelCallers(callGraph, stubFuncName);
            for (const auto &caller : highestCallers) {
                if (caller == stubFuncName) {
                    continue;
                } else {
                    functionCallersMap[caller] = "caller";
                }
            }
        }

        for (const auto &entry : functionCallersMap) {
            runKLEECommand(entry.first, entry.second, modifiedFilePath, OutputDir);
        }
    } else {
        std::cerr << "Unsupported input file type. Please provide a .json or .bc file.\n";
        return 1;
    }
    return 0;
}
