import numpy as np
import docker
import os
import sys
import tempfile

class InferenceSession:
    def __init__(self, model_path, **kwargs):
        if model_path.endswith(".mlir") :
            model_suffix = ".mlir"
        elif model_path.endswith(".onnx") :
            model_suffix = ".onnx"
        elif model_path.endswith(".so") :
            self.compiled_lib = os.path.abspath(model_path)
            self.session = self.getSession()
            return;
        else :
            print(
                    "Invalid input model path. Must end with .onnx or .mlir or .onnxtext"
                )
            exit(1)

        if "compile-options" in kwargs.keys():
            self.compile_options = kwargs["compile-options"]
        else:
            self.compile_options = ""
            

        if "onnx-mlir-container" in kwargs.keys():
            self.compiler_container = kwargs["onnx-mlir-container"]
        else:
            # Default image
            # The compiler command may have different path in different image
            #self.onnx_mlir_image = "ghcr.io/onnxmlir/onnx-mlir-dev" 
            self.onnx_mlir_image = "onnxmlir/onnx-mlir-dev"
         

        # Path to mount the model to the image
        self.container_model_dirname = "/myspace"
        self.container_output_path = "/myoutput"

        self.model_path = model_path

        # Construct compilation command

        # Assume we are using onnx-mlir-dev container
        command_str = "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"
        absolute_path = os.path.abspath(self.model_path)
        self.model_basename = os.path.basename(absolute_path)
        self.model_dirname = os.path.dirname(absolute_path)
        
        # Compiled library
        command_str += " " + self.compile_options
        command_str += " " + os.path.join(self.container_model_dirname, self.model_basename)

        """
        # ToFix: should use temporary directory for compilation, and
        # use "-o" to put the compiled library in the temporary directory.
        #self.output_dirname = tempfile.TemporaryDirectory()
        #self.compiled_model = os.path.join(self.output_dirname.name, self.model_basename.removesuffix(model_suffix)+".so")
        command_str += " -o " + self.container_output_path
        """
        self.output_dirname = self.model_dirname
        self.compiled_model = os.path.join(self.output_dirname, self.model_basename.removesuffix(model_suffix)+".so")

        print(command_str)
        print(self.compiled_model)

        self.container_client = docker.from_env()
        """
        msg=self.container_client.containers.run(self.onnx_mlir_image,
            command_str,
            volumes={self.model_dirname: {'bind': self.container_model_dirname, 'mode': 'r'}, self.output_dirname: {'bind': self.container_output_path, 'mode': 'rw'}
            }
        )
        """
        msg=self.container_client.containers.run(self.onnx_mlir_image,
            command_str,
            volumes={self.model_dirname: {'bind': self.container_model_dirname, 'mode': 'rw'}
            }
        )
        print(msg)
        self.session = self.getSession()

    def getSession(self):
        if not os.environ.get("ONNX_MLIR_HOME", None):
            raise RuntimeError(
                "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to " 
                "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
            "th    e parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
            "execu    tables and libraries can be found, typically `onnx-mlir/build/Debug`" 
            )      
        RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
        print(RUNTIME_DIR)
        sys.path.append(RUNTIME_DIR)
        try:    
            from PyRuntime import OMExecutionSession
        except ImportError:
            raise ImportError(
                "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
                "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
            )
        return OMExecutionSession(self.compiled_model, "NONE")

    def run(self, outputname, inputs, **kwargs):
        return self.session.run(inputs)
