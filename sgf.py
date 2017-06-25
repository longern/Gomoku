from os import listdir
import codecs
import re

def read_sgf():
	sgf_files = []
	pattern = re.compile("\[[a-o]{2}\]")
	for filename in listdir("sgf"):
		content = codecs.open("sgf/" + filename, "r", "gbk").read()
		history = []
		for chess in re.findall(pattern, content):
			history.append((ord(chess[1]) - ord('a'), ord(chess[2]) - ord('a')))
		if history:
			sgf_files.append(history)
	return sgf_files
