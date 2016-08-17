from  CMS_SURF_2016.utils.archiving import KerasTrial
import os, sys

def addCommitPushDir(folder, remote="origin", branch="master"):
	out1 = os.popen("git -C %r add ." % folder)
	print(out1)
	out2 = os.popen("git -C %r commit -a -m 'added %r'" % (folder, folder))
	print(out2)
	out3 = os.popen("git -C %r push %r %r" % (folder, remote, branch))
	print(out3)

def commitAllTrials(archive_dir,remote="origin", branch="master"):
	paths = KerasTrial.get_all_paths(archive_dir)
	for path in paths:
		addCommitPushDir(path,remote="origin", branch="master")
