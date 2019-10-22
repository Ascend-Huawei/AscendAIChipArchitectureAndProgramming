#!/bin/sh

omg --model=model/deploy_mylenet-1.prototxt --weight=model/mylenet-1.caffemodel --framework=0 --plugin_path=plugin --output=mylenet --ddk_version=1.3.T21.B880
