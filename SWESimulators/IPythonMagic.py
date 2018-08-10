# -*- coding: utf-8 -*-

"""
This python module implements helpers for IPython / Jupyter and CUDA

Copyright (C) 2018  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from IPython.core.magic import line_magic, Magics, magics_class
import pycuda.driver as cuda



@magics_class
class CudaContextHandler(Magics): 
    @line_magic
    def cuda_context_handler(self, context_name):
        print("Registering " + context_name + " as a global context")
        
        if context_name in self.shell.user_ns.keys():
            print("`-> Context already registered! Ignoring")
            return
        else:
            print("`-> Creating context")
            self.shell.ex(context_name + " = Common.CudaContext(verbose=True, blocking=False)")
        
        # this function will be called on exceptions in any cell
        def custom_exc(shell, etype, evalue, tb, tb_offset=None):
            print("Exception caught: Resetting to CUDA context " + context_name)
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                print("`-> popping " + str(context.handle))
                cuda.Context.pop()

            if context_name in self.shell.user_ns.keys():
                print("`-> pushing " + str(self.shell.user_ns[context_name].cuda_context.handle))
                self.shell.ex(context_name + ".cuda_context.push()")
            else:
                print("No CUDA context called " + context_name + " found (something is wrong)!")
                print("CUDA will not work now")

            # still show the error within the notebook, don't just swallow it
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

        # this registers a custom exception handler for the whole current notebook
        get_ipython().set_custom_exc((Exception,), custom_exc)
        
        
        # Handle CUDA context when exiting python
        import atexit
        def exitfunc():
            print("Exitfunc: Resetting CUDA context stack")
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                print("`-> popping " + str(context.handle))
                cuda.Context.pop()
        atexit.register(exitfunc)

print("Registering automatic CUDA context handling")
print("(use %cuda_context_handler my_context to create a context called my_context")
ip = get_ipython()
ip.register_magics(CudaContextHandler)
