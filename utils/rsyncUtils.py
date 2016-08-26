'''Automates rsyncing archive information between machines'''
from  CMS_SURF_2016.utils.archiving import KerasTrial, split_hash
import os, sys

def rsyncStorable(hashcode, src_archive_dir, dest_archive_dir):
	if(src_archive_dir[-1] != "/"): src_archive_dir = src_archive_dir + "/"
	if(dest_archive_dir[-1] != "/"): dest_archive_dir = dest_archive_dir + "/"

	rel_path = "blobs/" + "/".join(split_hash(hashcode)) + "/"
	print(rel_path)

	out1 = os.popen("cd %s && rsync -a --relative %r %r" % (src_archive_dir,rel_path, dest_archive_dir)).read()
	

def commitAllTrials(archive_dir,remote="origin", branch="master"):
	paths = KerasTrial.get_all_paths(archive_dir)
	for path in paths:
		addCommitPushDir(path,remote="origin", branch="master")

rsyncStorable('f3b48f37e401891083aac927f86375f2e96015ac', "MyArchiveDir/", "dweitekamp@titans.hep.caltech.edu:~/fake_archive_dir/")
