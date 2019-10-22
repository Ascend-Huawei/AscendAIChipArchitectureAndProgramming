# Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
#
ll : libcaffe_reduction_layer.so lib_caffe_parser.so

#blank line is not allowed in makefile

ifeq ($(DDK_PATH),)
TOPDIR := $(HOME)/tools/che/ddk/ddk
else
TOPDIR := $(DDK_PATH)
endif

CC := LD_LIBRARY_PATH=$(TOPDIR)/uihost/lib:$$LD_LIBRARY_PATH g++

LOCAL_DIR  := ./
INCLUDE_DIR := $(TOPDIR)/include

SRC_DIR = $(LOCAL_DIR)
BUILD_DIR = tmp
OBJ_DIR = $(BUILD_DIR)/obj
DEPS_DIR  = $(BUILD_DIR)/deps

#add include file
INC_DIR = \
    -I$(SRC_DIR)\
    -I$(INCLUDE_DIR)/inc\
    -I$(INCLUDE_DIR)/inc/custom\
    -I$(INCLUDE_DIR)/inc/graph\
    -I$(INCLUDE_DIR)/third_party/protobuf/include\
    -I$(INCLUDE_DIR)/third_party/json/include\
    -I$(INCLUDE_DIR)/libc_sec/include \
    -I/usr/include/python2.7

#add compile options
CC_FLAGS := $(INC_DIR) -g -std=c++11 -fPIC
LNK_FLAGS := \
    -L/usr/lib/python2.7/config-x86_64-linux-gnu -lpython2.7\
	-L$(TOPDIR)/uihost/lib -lomg -lte_fusion \
    -shared

DEMO_LNK_FLAGS := \
    -L/usr/lib/python2.7/config-x86_64-linux-gnu -lpython2.7\
    -shared

#Recursively traverse the 3 level subdirectory
DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)
CUSTOM_DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)
# caffe.proto exist,add inc/custom/omg/parser/caffe as source file
CAFFE_PROTO_FILE = $(LOCAL_DIR)/proto/caffe/caffe.proto
ifeq ($(CAFFE_PROTO_FILE), $(wildcard $(CAFFE_PROTO_FILE)))
DIRS += $(INCLUDE_DIR)/inc/custom/omg/parser/caffe
endif
# DDK_PATH is not defined,add add inc/custom/omg/parser/caffe as source file
ifeq ($(DDK_PATH),)
DIRS += $(INCLUDE_DIR)/inc/custom/omg/parser/caffe
endif
#Add each subdirectory to the search path.
VPATH = $(DIRS)

#search all .cpp file in  sub_dir
SOURCES  = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
CUSTOM_SOURCES  = $(foreach dir, $(CUSTOM_DIRS), $(wildcard $(dir)/*.cpp))


OBJS   = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(SOURCES))))
OBJS_customop = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(CUSTOM_SOURCES))))
OBJS_no_customop := $(filter-out $(OBJS_customop), $(OBJS))
DEPS  = $(addprefix $(DEPS_DIR)/, $(patsubst %.cpp,%.d,$(notdir $(SOURCES))))

# libcaffe_reduction_layer.so only include the custom op code
# lib_caffe_parser.so include caffe.proto and caffe parser code ,not include custom op code

libcaffe_reduction_layer.so: $(OBJS_customop)
	$(CC) -c $(CC_FLAGS) -o proto/caffe/caffe.pb.o proto/caffe/caffe.pb.cc
	$(CC) $^ $(LNK_FLAGS) -o $@

lib_caffe_parser.so: $(OBJS_no_customop)
	$(CC) -c $(CC_FLAGS) -o proto/caffe/caffe.pb.o proto/caffe/caffe.pb.cc
	@if [ -f $(LOCAL_DIR)/proto/caffe/caffe.proto ]; then $(CC) $^ proto/caffe/caffe.pb.o $(LNK_FLAGS) -o $@; fi;

#before compile, make sure the OBJ_DIR is existing
$(OBJ_DIR)/%.o:%.cpp
	@if [ ! -d $(LOCAL_DIR)/proto/caffe ]; then mkdir -p $(LOCAL_DIR)/proto/caffe; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/caffe/caffe.pb.h ]; then if [ -f $(LOCAL_DIR)/proto/caffe/caffe.proto ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/caffe --proto_path=$(LOCAL_DIR)/proto/caffe $(LOCAL_DIR)/proto/caffe/caffe.proto; else LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/caffe --proto_path=$(INCLUDE_DIR)/inc/custom/proto/caffe $(INCLUDE_DIR)/inc/custom/proto/caffe/caffe.proto; fi; fi;
	@if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi;
	$(CC) -c $(CC_FLAGS) -o $@ $<

#before compile, make sure the DEPS_DIR is existing
$(DEPS_DIR)/%.d:%.cpp
	@if [ ! -d $(LOCAL_DIR)/proto/caffe ]; then mkdir -p $(LOCAL_DIR)/proto/caffe; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/caffe/caffe.pb.h ]; then if [ -f $(LOCAL_DIR)/proto/caffe/caffe.proto ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/caffe --proto_path=$(LOCAL_DIR)/proto/caffe $(LOCAL_DIR)/proto/caffe/caffe.proto; else LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/caffe --proto_path=$(INCLUDE_DIR)/inc/custom/proto/caffe $(INCLUDE_DIR)/inc/custom/proto/caffe/caffe.proto; fi; fi;
	@if [ ! -d $(DEPS_DIR) ]; then mkdir -p $(DEPS_DIR); fi;
	set -e; rm -f $@;
	$(CC) -MM $(CC_FLAGS) $< > $@.$$$$;
	sed 's,\($*\)\.o[ :]*,$(OBJ_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@;
	rm -f $@.$$$$

#ignore the error by adding "-" before the command
ifneq ($(MAKECMDGOALS), clean)
	-include $(DEPS)
endif

.PHONY : clean
clean:
	rm -rf $(BUILD_DIR) lib*.so proto/*.pb.* proto/caffe/*.pb.*
