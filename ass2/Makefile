FILES = Assignment2.m LoadBatch.m EvaluateClassifier.m ComputeCost.m MiniBatchGD.m

all: merged_code.m report.pdf


report.pdf: report.md
	pandoc report.md -o report.pdf


merged_code.m : $(FILES)
	@for filename in $(FILES); do					\
	    echo "\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";	\
	    echo  "%%Content of file : $$filename";			\
	    echo  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n";	\
	    cat "$$filename";						\
	done > merged_code.m ;
#cat  $(FILES) > merged_code.m
