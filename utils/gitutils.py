'''Automates pushing archive information to a remote git repo'''
from  CMS_SURF_2016.utils.archiving import KerasTrial
import os, sys

def addCommitPushDir(folder, remote="origin", branch="master"):
	out1 = os.popen("cd %s && git add ." % (folder)).read()
	print(out1)
	out2 = os.popen("cd %s && git commit -a -m 'added %r'" % (folder, folder)).read()
	if(not "nothing added to commit" in out2): print(out2)
	out3 = os.popen("cd %s && git push %r %r" % (folder, remote, branch)).read()
	print(out3)

def commitAllTrials(archive_dir,remote="origin", branch="master"):
	paths = KerasTrial.get_all_paths(archive_dir)
	for path in paths:
		addCommitPushDir(path,remote="origin", branch="master")
