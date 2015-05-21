
all : .hello
RR = R --vanilla --slave --quiet

.hello : ../Processed/train_features.csv \
      ../Processed/test_features.csv \
      ../Processed/train_scrubbed.csv
	$(RR) < hello_world_model.r

