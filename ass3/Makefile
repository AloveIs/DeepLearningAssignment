FILES = Assignment3.m ComputeCost.m ComputeGradients.m EvaluateClassifier.m LoadBatch.m MiniBatchGD.m
SHELL := /bin/bash

all: merged_code.m report.pdf


report.pdf: report.md
	cd pictures;	source script.sh; cd ..;
	pandoc report.md -o report.pdf;


merged_code.m : $(FILES)
	@for filename in $(FILES); do					\
	    echo "\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";	\
	    echo  "%%Content of file : $$filename";			\
	    echo  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n";	\
	    cat "$$filename";						\
	done > merged_code.m ;
#cat  $(FILES) > merged_code.m
