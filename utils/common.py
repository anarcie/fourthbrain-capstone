import sys
import hashlib
import os
import inspect
import re
import shutil

VERBOSE_OUTPUT = False
LINEWIDTH = 999


# --------------------------------------
# init
# --------------------------------------
#
#
# --------------------------------------
def init(verbose=False):
    '''initialize the verbos output, depreciated

    Args:
        verbose (bool): sets verbose state

    Returns:
        None

    '''
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = verbose


# --------------------------------------
# OutList
# --------------------------------------
#
#
# --------------------------------------
def OutList(OList, Lvl=1, isVerbose=False):
    '''Wrapper of the `Out` function, this will handle
    outputting a list in the same format as `Out`

    Args:
        Olist (list): List to be printed out
        Lvl (int): Level of indentation, depreciated
        isVerbose (bool): if the list is only for verbose mode

    Returns:
        None

    '''
    if (isVerbose and VERBOSE_OUTPUT) or not isVerbose:
        for item in OList:
            Out(item, Lvl, isVerbose)


# --------------------------------------
# Out
# --------------------------------------
# Manage Output, Handles Verbosity and
# Indents
# --------------------------------------
def Out(Msg, Lvl=1, isVerbose=False, header=False, src=False):
    '''Handles all screen outputs and logging

    Args:
        Msg (str): message to output to screen
        Lvl (int): depreciated
        isVerbose (bool): Marks the output as verbose mode only
        header (bool): depreciated
        src (bool): depreciated

    Returns:
        None

    '''
    # Set the Line Width minus 1 for 0 index of chars
    LWidth = LINEWIDTH - 1
    try:
        Msg = str(Msg)
        stack = inspect.stack()
        stackMax = len(stack)
        stackLength = stackMax - 3
        if src is not False:
            stackLength = stackLength - 1
        Caller = src if src is not False else stack[1][3]
        # print('--------------------')
        # print(stackLength)
        # for i, r in enumerate(stack):
        #     print(i, stack[i][3])
        if (isVerbose and VERBOSE_OUTPUT) or not isVerbose:
            Indent = " " * ((stackLength - 1) * 4)
            if VERBOSE_OUTPUT is True:
                print(" %s[%s] %s" % (Indent, Caller, Msg[:LWidth]))
                if (len(Msg)) > LWidth:
                    Out(Msg[LWidth:], Lvl, isVerbose, header, Caller)
            else:
                print(" %s%s" % (Indent, Msg[:LWidth]))
                if (len(Msg)) > LWidth:
                    Out(Msg[LWidth:], Lvl, isVerbose, header, Caller)
    except Exception as e:
        print('Could Not common.Out: ')
        print(GeneralExceptionMessage(e))


# --------------------------------------
# ensureFolder
# --------------------------------------
#
#
# --------------------------------------
def ensureFolder(path):
    '''Ensures that a folder path exists, if it
    not, will create the folder path

    Args:
        path (str): Folder or File path to ensure

    Returns:
        bool: if the folder path now or previously existed (True)

    '''
    normpath = os.path.normpath(path)
    try:
        if not os.path.isdir(os.path.dirname(normpath)):
            os.makedirs(os.path.dirname(normpath))
            return True
        else:
            return True
    except Exception as e:
        print(e)
        return False


# --------------------------------------
# deletePath
# --------------------------------------
#
#
# --------------------------------------
def deletePath(path):
    '''Deletes a file or directory path

    Args:
        path (str): Directory or File path to be deleted

    Returns:
        bool: returns if this was successfull, if not, exits with a 1 exit code

    '''
    normpath = os.path.normpath(path)
    # Handle Files
    if os.path.isfile(normpath):
        try:
            Out(f'Deleting File: {normpath}')
            os.unlink(path)
            return True
        except Exception as e:
            Out(f'Could Not Delete File {normpath}')
            Exit(GeneralExceptionMessage(e))
            return False

    if os.path.isdir(normpath):
        try:
            Out(f'Deleting Directory: {normpath}')
            shutil.rmtree(normpath)
            return True
        except Exception as e:
            Out(f'Could Not Delete Directory {normpath}')
            Exit(GeneralExceptionMessage(e))
            return False


# --------------------------------------
# Exit
# --------------------------------------
#
#
# --------------------------------------
def Exit(ExitMessage="Finished", ExitCode=1):
    '''Handles exiting of the program, with 0 and 200 being 'clean'
    exit codes.

    Args:
        ExitMessage (str): Message to dispaly on exit
        ExitCode (int): Exit code to send to the OS / Caller

    Returns:
        None

    '''

    if ExitCode == 0:
        Out(ExitMessage)
        Out('Exiting with Successful Build', 2, False)
        # Inverse the Exit Code For Jenkins Stupid Build
        # System to Fail the build and not send email
        sys.exit(0)
        # raise SystemExit
    elif ExitCode == 200:
        Out('Ok Failure Exit (200): ' + str(ExitMessage), 0, False)
        sys.exit(200)
    else:
        Out('Error Encountered: ' + str(ExitMessage), 1, False)
        sys.exit(ExitCode)


# --------------------------------------
# GeneralExceptionMessage
# --------------------------------------
#
#
# --------------------------------------
def GeneralExceptionMessage(e):
    '''Handles pulling out the exception message from
    and except, displaying it and then exits the run

    Args:
        e (obj): error object

    Returns:
        None

    '''
    if hasattr(e, 'message'):
        Exit(e.message)
    else:
        Exit(e)


# --------------------------------------
# CheckExpectedItems
# --------------------------------------
#
#
# --------------------------------------
def CheckExpectedItems(items, expected):
    '''Checks a list for the expected length

    Args:
        items (list): the list to check
        expected (int): the len expected of the list

    Returns:
        bool: if the expected length matches the list length

    '''
    actual = len(items)
    if actual == expected:
        return True

    return False


# --------------------------------------
# ListRegex
# --------------------------------------
#
#
# --------------------------------------
def ListRegex(List, Regex):
    '''Preforms a regex on a list and then returns the
    matched list items

    Args:
        List (list): List to be checked
        Regex (str): Regex to use on the check

    Returns:
        list: if items are found a list is returned
        bool: if no items match, False is returned

    '''
    haystack = []
    r = re.compile(Regex)
    haystack = list(filter(r.match, List))
    if len(haystack) > 0:
        return haystack
    else:
        return False


# --------------------------------------
# CheckPath
# --------------------------------------
# Check for a File path & also its
# relative to script path
# --------------------------------------
def CheckFilePath(PassedPath, isFolder=False):
    '''Checks a file path and determines if it is
    relative to the script or absolute, returning the
    correct path

    Args:
        PassedPath (str): Path to check

    Returns:
        str: If the path exists regardless of relative or abs, returns absolute path
        bool: if the path cannot be found, returns False

    '''
    # Calculate Relative Path as Well
    sPath = os.path.dirname(os.path.abspath(__file__))
    ScriptPath = os.path.normpath(sPath)
    NormalizedPath = os.path.normpath(PassedPath)
    NormalizedRel = os.path.abspath(
        os.path.join(ScriptPath, '..', NormalizedPath))

    # Check Absolute Path
    Out('Checking absolute: ' + NormalizedPath, 1, True)

    if isFolder is False:
        if os.path.isfile(NormalizedPath):
            return NormalizedPath

        # Check Relative Path
        Out('Checking relative: ' + NormalizedRel, 1, True)
        if os.path.isfile(NormalizedRel):
            return NormalizedRel

    else:
        if os.path.isdir(NormalizedPath):
            return NormalizedPath

        # Check Relative Path
        Out('Checking relative: ' + NormalizedRel, 1, True)
        if os.path.isdir(NormalizedRel):
            return NormalizedRel

    return False


# --------------------------------------
# GenerateHash
# --------------------------------------
# Used to Tell if we need to do a deploy
# StackOver Flow
# https://stackoverflow.com/a/49701019
# --------------------------------------
def GenerateHash(path):
    '''Generates a Hash based on the input of a directory tree

    Args:
        path (str): Path to be hashed

    Returns:
        str: Hash of the given directory

    '''
    digest = hashlib.sha1()

    for root, dirs, files in os.walk(path):
        for names in files:
            file_path = os.path.join(root, names)

            # Hash the path and add to the digest
            # to account for empty files/directories
            digest.update(hashlib.sha1(
                file_path[len(path):].encode()).digest())

            # Per @pt12lol - if the goal is uniqueness over
            # repeatability, this is an alternative method using 'hash'
            # digest.update(str(hash(file_path[len(path):])).encode())

            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f_obj:
                    while True:
                        buf = f_obj.read(1024 * 1024)
                        if not buf:
                            break
                        digest.update(buf)

    return digest.hexdigest()


def getOneLevelUp(filepath):
    '''Gets the file path 1 level up from the passed path.

    Args:
        filepath (str): Path to be moved 1 up (to parent directory)

    Returns:
        str: new file path string

    '''
    path = os.path.abspath(os.path.normpath(filepath))
    joinPath = os.path.join(path, '../')
    upOneLevel = os.path.abspath(joinPath)
    return upOneLevel


# --------------------------------------
# main
# --------------------------------------
#
#
# --------------------------------------
def main(sysarg):
    '''Main function. You should not be here!

    Args:
        sysarg (list): List of passed args

    Returns:
        None

    '''
    Exit('Not Built For Calling Directly')


if __name__ == "__main__":
    main(sys.argv)
