	??릔;C@??릔;C@!??릔;C@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??릔;C@?~߿yq@1?{?ʄ?1@A.?+=)??IB`??"C2@rEagerKernelExecute 0*	      T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?W[?????!?`????B@)HP?sג?1?m۶m?6@:Preprocessing2U
Iterator::Model::ParallelMapV2;?O??n??!.>9\6@);?O??n??1.>9\6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?&S???!?Mozӛ6@)?HP???1?'?ˀO.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??0?*??!????Q-@)??0?*??1????Q-@:Preprocessing2F
Iterator::Modelc?ZB>???!۶m۶m?@)?<,Ԛ?}?1~?:?""@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!&??RL?@)?~j?t?x?1&??RL?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz?,C??!I?$I?$Q@)n??t?1?,d!Y@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?0?*???!|r??8@)ŏ1w-!_?1?p???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?47.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???`9'K@Q}%???F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~߿yq@?~߿yq@!?~߿yq@      ??!       "	?{?ʄ?1@?{?ʄ?1@!?{?ʄ?1@*      ??!       2	.?+=)??.?+=)??!.?+=)??:	B`??"C2@B`??"C2@!B`??"C2@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???`9'K@y}%???F@