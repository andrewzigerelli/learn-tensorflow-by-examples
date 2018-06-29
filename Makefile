
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

TRAIN := toxic_textcnn_train.py
INFER = toxic_textcnn_infer.py

TXTFILES = test.txt

.IGNORE: profile check train

MODEL_FILENAME = output/inception-v3.meta
CONV_SCRIPT = ./inception-v3.py

INPUT_NODE_FLAG = -in=input
OUTPUT_NODE_FLAG = -on=InceptionV3/Predictions/Reshape_1

.PHONY: all
all: check
#all: profile check

.PHONY: train
train : $(TRAIN)

$(TRAIN) : weight.pkl

%.pkl: 
	@if [ ! -e $@	]; then \
	    rm -rf graphs/; \
	    # python $(TRAIN) 2>&1 | grep -v "I tensorflow"; \
	fi
	    python $(TRAIN); \

.PHONY: infer
infer: train inference

inference : $(INFER)
	python $(INFER)

.PHONY: prereqs
prereqs:
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py

.PHONY: profile
profile: weights
	${NCPROFILE} -s 4 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: browse_profile
browse_profile: weights
	${NCPROFILE} -s 4 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}
	firefox output_report.html &

.PHONY: weights
weights:
	@sed -i 's/\r//' ${CONV_SCRIPT}
	@chmod +x ${CONV_SCRIPT}
	test -f ${MODEL_FILENAME} || (${GET_WEIGHTS} && ${CONV_SCRIPT})

.PHONY:
compile: weights
	test -f graph || ${NCCOMPILE} -s 4 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: check
check: weights
	-${NCCHECK} -s 4 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG} -i ../../data/images/cat.jpg -id 917 -M 128 -S 2 -cs 0,1,2 -metric top1

.PHONY: run
run:
	./run.py

.PHONY: run_py
run_py:
	./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, check, cpp, run_py, run_cpp";
	@echo "  make weights - downloads the trained model";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser.";
	@echo "  make run_py - runs the run.py python example program";
	@echo "  make clean - removes all created content"

clean:
	rm weight.pkl
	rm -rf graphs/
