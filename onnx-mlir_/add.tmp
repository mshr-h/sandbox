builtin.module  {
  builtin.func @main_graph(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {input_names = ["Input1", "Input2"], output_names = ["Output"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1] , \22name\22 : \22Input1\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1] , \22name\22 : \22Input2\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1] , \22name\22 : \22Output\22 }\0A\0A]\00"} : () -> ()
}
