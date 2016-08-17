from  CMS_SURF_2016.utils.archiving import KerasTrial
import os, sys

def addCommitPushDir(folder, remote="origin", branch="master"):
	out1 = os.popen("git --git-dir=%s.git --work-tree=%s add ." % (folder,folder)).read()
	print(out1)
	out2 = os.popen("git --git-dir=%s.git --work-tree=%s commit -a -m 'added %r'" % (folder,folder, folder)).read()
	print(out2)
	out3 = os.popen("git --git-dir=%s.git --work-tree=%s push %r %r" % (folder,folder, remote, branch)).read()
	print(out3)

def commitAllTrials(archive_dir,remote="origin", branch="master"):
	paths = KerasTrial.get_all_paths(archive_dir)
	for path in paths:
		addCommitPushDir(path,remote="origin", branch="master")
