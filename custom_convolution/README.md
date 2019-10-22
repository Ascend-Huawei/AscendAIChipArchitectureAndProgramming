插件编译

1）修改 custom_xxx/plugin/ 的 Makefile 文件中的 DDK 路径（位于第12行）。

2）修改 custom_xxx/omg_verify/env_omg.sh 的 DDK_PATH，并执行 source env_omg.sh。

3）在 custom_xxx/plugin/ 目录下执行 make clean; make。

OMG（模型转换） 下面介绍通过命令行方式进行 OMG（模型转换）。

1）修改 custom_xxx/omg_verify/env_omg.sh 的 DDK_PATH。

2）修改 omg.sh 的 DDK_PATH 与 ddk_version 参数。

3）在 custom_xxx/omg_verify/ 目录下依次执行 source env_omg.sh 与 bash omg.sh。
