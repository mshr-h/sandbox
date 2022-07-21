# WebAssembly with Zig

```bash
mkdir wasm-sample
cd wasm-sample
zig init-exe
zig build-exe src/main.zig -target wasm32-wasi
wasmtime main.wasm
```
