# WASM Stuff

## Run WebAssembly binary using wasmtime

### Prerequisites

- Rustup
- Wasmtime

### add `wasm32-wasi` target 

```bash
rustup target add wasm32-wasi
```

### build rust code into `.wasm`

```bash
rustc hello.rs --target wasm32-wasi
```

### run wasm

```bash
wasmtime hello.wasm
```

