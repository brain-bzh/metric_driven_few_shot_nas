from copy import deepcopy
from pathlib import Path
from .paths import *

class Output():
    #Generic class for handling and saving outputs of experiments
    #In online mode, output file is updated during experiment runtime
    #In offline mode, output file is created once upon closing the output

    def __init__(self, name, out_var, mode='online', separator=';'):

        self._name = name
        self._mode = mode
        self._separator = separator
        self._var = out_var
        self._path = OUTPUTS_PATH + '/' + self._name +'.csv'

        self._vardict = dict.fromkeys(self._var, None)
        if self._mode == 'offline':
            self._vardict_history = []

        if self._mode == 'online':                                      #New file is only created at initialization in online mode

            out = open(self._path, 'a+')
            out.write(self._separator.join(self._var) + self._separator + '\n')
            out.close()
    
    def add_output(self, var, value):
        #Add new value to the selected variable
        #This can be done in any order, definition order will be retained when writing to the file

        self._vardict[var] = value

    def write_output(self):
        #Create a new line for the output
        #If in online mode, the line is immediately written to the file

        if self._mode == 'online':
            out = open(self._path, 'a+')
            out.write(self._separator.join([str(self._vardict[var]) for var in self._var]) + self._separator + '\n')
            out.close()

        elif self._mode == 'offline':
            self._vardict_history.append(deepcopy(self._vardict))

        self._vardict = dict.fromkeys(self._var, None)

    def close(self):
        #Creates the file if in offline mode

        if self._mode == 'offline':
            out = open(self._path, 'a+')
            out.writelines([self._separator.join(self._var) + self._separator + '\n'] + [self._separator.join([str(self._vardict_history[i][var]) for var in self._var]) + self._separator + '\n' for i in range(len(self._vardict_history))])
            out.close()
        
