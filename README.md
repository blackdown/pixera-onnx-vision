# ONNX Vision for PIXERA

A RenderBridge extension that runs an ONNX vision model over a Live Input and
returns the result to the timeline, inside PIXERA's own process. A matting
model becomes a layer's alpha, giving a live key with no green screen; a depth
model gives a depth pass; an image-to-image model repaints the feed.

Inference runs on NVIDIA TensorRT. The extension is a single 685 KB file.

> [!WARNING]
> **Proof of concept — not for live production.**
>
> This is an experimental extension, shared to show what is possible. It has
> had limited testing, is not supported, and can fail in ways that affect
> playback. Try it on a machine that is not running a show, and do not rely on
> it in a live production environment.

---

## Contents

- [Requirements](#requirements)
- [Installing](#installing)
- [Using it](#using-it)
- [Which models work](#which-models-work)
- [Performance](#performance)
- [Caveats](#caveats)
- [Debugging](#debugging)
- [Building from source](#building-from-source)
- [Third-party components](#third-party-components)

---

## Requirements

- **An NVIDIA GPU.** TensorRT is NVIDIA-only; AMD and Intel GPUs cannot run
  this. Developed on Ada (RTX 4070) and Blackwell (RTX PRO 6000).
- **An NVIDIA driver new enough for TensorRT 10.** The extension uses the CUDA
  library that comes with the graphics driver, so there is **no CUDA toolkit
  to install**.
- **TensorRT 10**, installed once on the machine.
- **Enough VRAM** for the model — and rather more while its engine is being
  built than while it runs.

Apart from TensorRT, nothing needs to be installed alongside the extension:
it imports only Windows system libraries.

---

## Installing

Follow these in order. Steps 1 to 4 put files and settings in place; nothing
takes effect until the restart in step 5.

### 1. Check the NVIDIA driver

Check it — but do not update it reflexively; see the note below.

```
nvidia-smi
```

or NVIDIA Control Panel → *Help* → *System Information*.

Known to work:

| Driver | Where |
|---|---|
| **596.72** | PIXERA machines with Blackwell GPUs, as shipped |
| **610.88** | RTX 4070 laptop used in development |

**PIXERA's Blackwell machines ship with 596.72, which is new enough — there
is nothing to do here.** A show machine running an older driver than a
workstation is normal, not a problem to fix.

If a driver is too old, it fails visibly: TensorRT does not load and the log
says so. The practical test is therefore to carry on and read the log at
step 6.

> [!IMPORTANT]
> **Do not update the driver on a commissioned show machine just for this.**
>
> The graphics driver on a media server is part of a validated configuration.
> Capture cards, output timing, genlock and sync can all change with it, and
> some hardware ships with the driver version deliberately pinned. A driver
> update is a decision about the whole machine, not about this extension — and
> it should be followed by testing the whole output chain.

#### About the CUDA version `nvidia-smi` reports

The header shows something like `CUDA UMD Version: 13.3`. That is **the newest
CUDA the driver can run** — not something installed, and not a version you
need to match. A higher number is fine: CUDA drivers are backward compatible,
so a driver reporting 13.3 runs software built for CUDA 12.

**No CUDA toolkit is needed at all.** Where step 2 says "TensorRT for CUDA
12", that chooses which TensorRT download to take; it is not an instruction to
install CUDA 12.

For a fresh workstation rather than a show machine, a current Studio driver
from <https://www.nvidia.com/download/index.aspx> is the right choice.

### 2. Install TensorRT

Download TensorRT 10 for CUDA 12 from
<https://developer.nvidia.com/tensorrt/download/10x>.

Create a folder to keep it in — `C:\tensorrt` works well — and copy the
contents of the download's `bin` folder into it. The files that matter:

```
nvinfer_10.dll
nvinfer_plugin_10.dll
nvonnxparser_10.dll
nvinfer_builder_resource_sm<NN>_10.dll     one per GPU architecture
```

Choose the folder deliberately: moving it later means redoing step 3.

#### Which builder resources to keep

There is one builder resource per GPU architecture, and together they are
most of the download — about 2.8 GB. Only the one matching the card in the
machine is ever used.

| File | Architecture | Cards | Size |
|---|---|---|---|
| `sm75` | Turing | RTX 20-series, GTX 16-series, Quadro RTX 4000–8000, T4 | 157 MB |
| `sm80` | Ampere (datacenter) | A100, A30 | 251 MB |
| `sm86` | Ampere | RTX 30-series, RTX A2000–A6000, A40, A10 | 237 MB |
| `sm89` | **Ada Lovelace** | **RTX 40-series, RTX 6000 / 5000 / 4500 / 4000 Ada**, L40, L4 | 249 MB |
| `sm90` | Hopper (datacenter) | H100, H200 | 637 MB |
| `sm100` | Blackwell (datacenter) | B100, B200, GB200 | 402 MB |
| `sm120` | **Blackwell** | **RTX PRO 6000 Blackwell, RTX 50-series** | 368 MB |
| `ptx` | fallback | anything with no resource above | 472 MB |

The two in bold are the ones media servers use. Keep `sm86` for older Ampere
machines and `sm75` for Turing-era ones. The datacenter entries are for cards
with no display outputs and can always be removed. Keeping just `sm89` and
`sm120` covers current PIXERA hardware in 617 MB.

Note that **`sm120`, not `sm100`, is the workstation Blackwell** — the lower
number is the datacenter part.

`ptx` is used when no matching resource is present. It works, but compiles for
the architecture at build time, so engine builds are markedly slower. Keep the
right `sm` file rather than relying on it.

Unsure which card is which? Install one, load a model, and read the log: the
engine file name ends in the architecture the extension asked for, such as
`..._sm120.trtengine`.

### 3. Tell Windows where TensorRT is

The recommended way is to add the folder to the **system `PATH`**:

1. Press **Start** and type `environment`.
2. Choose **Edit the system environment variables**.
3. Click **Environment Variables…** at the bottom of the dialog.
4. In the **lower** box — *System variables*, **not** the upper *User
   variables* box — select **`Path`** and click **Edit…**.
5. Click **New** and enter the folder from step 2, for example `C:\tensorrt`.
   Enter the folder itself, with no trailing backslash.
6. Click **OK** on all three dialogs. Closing any of them with *Cancel*
   discards the change.

It must be a **system** variable. A user variable is not inherited by
processes started outside that user's session — including PIXERA when a
service or startup task launches it.

> [!CAUTION]
> **Do not use `setx` to add to `PATH`.** It truncates the value at 1024
> characters, and a system `PATH` is often longer. The result is a silently
> shortened `PATH` and a machine that has lost entries it needed. Use the
> dialog above.

#### Alternatively, `TENSORRT_DIR`

The extension also looks in the folder named by a `TENSORRT_DIR` system
environment variable, and that alone is enough. Set it in the same dialog:
under **System variables** click **New…**, name `TENSORRT_DIR`, value
`C:\tensorrt`. Setting both is harmless.

`PATH` is recommended because it is the standard mechanism, and because the
check below can confirm it.

#### Checking PATH

Open a **new** terminal — one already open will not see the change — and run
one of these.

In **Command Prompt**:

```
where nvinfer_10.dll
```

In **PowerShell** the `.exe` is required, because a bare `where` is an alias
for PowerShell's `Where-Object` and does something else entirely:

```powershell
where.exe nvinfer_10.dll
```

Each prints the full path when `PATH` is right. If Command Prompt answers
`INFO: Could not find files for the given pattern(s).`, the folder is not on
`PATH`.

### 4. Put the extension where it will live

Download `ExtOnnxVision.dll` from this repository's
[Releases](../../releases) page and copy it to wherever it will stay
permanently.

Do this *before* adding it in PIXERA. PIXERA records the path you browse to,
so a DLL that is later moved or renamed leaves a resource pointing at nothing.

### 5. Restart

Environment variables are read when a process starts, and one set in step 3
does not reach a PIXERA launched from an Explorer session that was already
running. Restarting PIXERA is often enough; restarting the machine always is.

### 6. Add the extension in PIXERA

Right-click in the resource browser, add a **generic extension**, and browse
to the DLL from step 4.

To confirm the machine is set up, add a layer and set a model as described
below. The first log line names the TensorRT it found and the file it came
from — see [Debugging](#debugging).

---

## Using it

1. **Add a fresh layer** using the extension. PIXERA stores a layer's
   parameter list in the project, so a layer created with a different version
   of the extension shows that version's parameters.

2. **Drag a Live Input Resource onto the layer's `Source` parameter.** That is
   what routes video into the extension. With no model set you should see
   correct video straight away — confirming the routing works before any model
   is involved.

3. **Type or paste the full path to an `.onnx` file into `Model File`.**

4. **Wait.** The first use of a model on a given GPU compiles an engine, which
   takes minutes. Video passes through untouched meanwhile, and `Status`
   counts the time.

### Parameters

| Parameter | What it does |
|---|---|
| `Source` | the Live Input, or any Resource, to process |
| `Mode` | *Process* is the working mode; *Idle* and *Passthrough (raw)* are diagnostics |
| `Swap Red/Blue` | corrects the channel order PIXERA delivers; leave it on |
| `Input Range` | how pixels are scaled before the model sees them |
| `Output Range` | how to interpret the result; *Auto* is usually right |
| `Result As` | *Alpha (key)* to key, *Greyscale (view)* to see the result itself, or *Both* |
| `Input Size Override` | for models whose input size is not fixed |
| `TensorRT Build Effort` | 0–5; lower compiles much faster and runs slightly slower |
| `Model File` | full path to an `.onnx` file |
| `Reload Model` | flip to reload the model, or to retry a failed load |
| `Status` | read-only: what the extension is doing, and the frame time |

Four more read-only outputs report what the model found rather than what it
drew: `Subject Coverage`, `Subject Centre X`, `Subject Centre Y` and
`Result Mean`.

### Choosing Input Range

This cannot be read from a model file; it depends on how the model was
trained. If a result looks like noise or is plainly wrong, change this first.

| Setting | Typical of |
|---|---|
| `ImageNet` | most segmentation and matting models |
| `0 to 1` | many depth models |
| `-1 to 1` | some generative and style models |
| `0 to 255` | style transfer models |

---

## Which models work

**Image in; image or single-channel map out.**

- **Single-channel output** — matting, segmentation, depth, saliency. The
  result becomes alpha, greyscale, or both.
- **Three-channel output** — style transfer, enhancement, colourisation. The
  result replaces the picture.

**Object detection models do not work.** Their output is a tensor of boxes and
scores that needs decoding written for that particular model. The extension
logs the output shape and passes frames through rather than showing noise.

**Every operator in the model must be supported by TensorRT.** There is no
fallback: a model using an operator TensorRT cannot build will not load, and
the log names the operator. Re-exporting the model from its original
framework can often avoid it.

**One input, one output.** Models with several inputs are not supported.

---

## Performance

Indicative figures from development testing, against a 1080p Live Input:

| Model | GPU | Inference | Full frame |
|---|---|---|---|
| BiRefNet 512 fp16 (matting) | RTX PRO 6000 Blackwell | 11 ms | 14 ms |
| BiRefNet 512 fp16 (matting) | RTX 4070 Laptop, 44 W | 40 ms | 46 ms |
| Depth Anything v2 small | RTX 4070 Laptop | 8–10 ms | ~20 ms |

*Full frame* adds reading the picture back from the GPU, preparing it, and
writing the result.

PIXERA expects a layer to update within about 50 ms. A laptop GPU running a
heavy model has little headroom and may stutter. Laptops are fine for trying
this out; the results above are why a workstation GPU is the realistic target.

### Engine compilation

TensorRT compiles each model into an engine optimised for the exact GPU, which
takes **minutes the first time** and is then cached in:

```
%LOCALAPPDATA%\ExtOnnxVision\trt-engines
```

Later loads of the same model take seconds. An engine is specific to the
model, the TensorRT version *and* the GPU architecture, so copying the cache to
a machine with a different card achieves nothing.

**Compile every model you intend to use, on the machine you will use it on,
before you need it.**

To experiment faster, set `TensorRT Build Effort` to `1`, then back to `3` for
the engine you keep.

### Rebuilding an engine

**`Reload Model` does not rebuild the engine.** It reloads the model and reuses
the engine compiled before, which is why it takes seconds.

Usually no rebuild is needed by hand: changing the model, `Input Size
Override`, `TensorRT Build Effort`, the GPU or the TensorRT version all change
the engine's name, so a new one is compiled automatically.

To force a rebuild with nothing else changed — after a driver update, or if an
engine is suspect — **delete the engine file, then flip `Reload Model`.** The
log names the file in use, so only that one needs removing:

```
engine file for this model, GPU and settings: C:\Users\<you>\AppData\Local\ExtOnnxVision\trt-engines\<model>_..._sm120.trtengine
```

There is deliberately no rebuild control on the layer. Every parameter PIXERA
shows can be keyframed and dragged, and a rebuild that takes minutes and
discards a working engine is not something a timeline should be able to
trigger.

---

## Caveats

In addition to the [proof of concept warning](#onnx-vision-for-pixera) above:

**The model path is remembered by the extension, not the project.** PIXERA
cannot store a text value on a layer, so the extension keeps the path itself,
against the layer's identity, and restores it when the project is reopened.
If the `.onnx` file has moved, the layer comes back with no model set.

**Do not drag the `Model File` box.** PIXERA shows every parameter as a
numeric control; dragging this one replaces the path with a number and unloads
the model. Type or paste the path.

**Compiling needs more VRAM than running.** A model that runs comfortably may
still fail to build with little memory free. Close other GPU work during a
first compile.

**Readback costs time.** Frames are copied from the GPU to the CPU and back
for every frame processed. On a fast GPU this becomes a noticeable share of
the frame time.

---

## Debugging

### Logs

The extension writes its own trace, unbuffered so it survives a crash — the
last line is the last step that completed:

```
%USERPROFILE%\Documents\ExtOnnxVision-log.txt
```

The same messages also reach PIXERA's RX log:

```
C:\ProgramData\AV Stumpfl\Pixera\logs\rx_log.txt
```

Only errors appear in PIXERA's main log, so that staying quiet is normal.

To follow progress live:

```powershell
Get-Content "$env:USERPROFILE\Documents\ExtOnnxVision-log.txt" -Wait -Tail 30
```

### A healthy start

```
TensorRT 10.15.1 from C:\tensorrt\nvinfer_10.dll (bound to the library's own API version, ...)
engine file for this model, GPU and settings: C:\Users\<you>\...\<model>_..._sm89.trtengine
compiling a TensorRT engine for this model and GPU; this takes minutes and only happens once
engine ready, input [1,3,512,512] NCHW -> output [1,1,512,512]
Ready on TensorRT 10.15.1, 10 ms/frame
```

On later loads the compile line is replaced by
`reusing the engine compiled earlier for this model and GPU`.

The first line is the one to check: it names the TensorRT actually loaded and
the file it came from.

### Common problems

| What you see | What it means |
|---|---|
| `nvinfer_10.dll not found…` | TensorRT is not installed, or its folder is on neither `PATH` nor `TENSORRT_DIR`. Check with `where.exe nvinfer_10.dll` in a new terminal, and restart PIXERA after changing either. |
| `nvcuda.dll could not be loaded…` | No NVIDIA GPU is in use, or no NVIDIA driver is installed. Check that `nvidia-smi` runs. |
| Engine build fails immediately after TensorRT loads | Often the builder resource for this GPU is missing from the TensorRT folder — see step 2 — or the driver is older than this TensorRT release expects. Read step 1 before updating a show machine's driver. |
| `Status` stays on `Building TensorRT engine` for minutes | Normal on a first load; video passes through meanwhile. Lower `TensorRT Build Effort` while experimenting. |
| `TensorRT could not parse this model…` | The model uses an operator TensorRT cannot build. The message names it. |
| `could not allocate GPU buffers…` | Not enough free VRAM. Close other GPU work and flip `Reload Model`. |
| `Load model from 0.01 failed` | The `Model File` parameter was dragged. Type the path again. |
| Video is blue | `Mode` is on *Passthrough (raw)*, which cannot correct channel order. Use *Process*. |
| Result looks like noise | `Input Range` is wrong for this model. Try the others. |
| Matte is inverted, or the whole frame is keyed | `Output Range` guessed wrong. Set it explicitly instead of *Auto*. |
| Parameters missing or wrong | The layer was created with a different version of the extension. Add a fresh layer. |
| Nothing at all in the log | The extension did not load. Check the DLL path in the resource. |

### Starting clean

Force every engine to be recompiled:

```powershell
Remove-Item "$env:LOCALAPPDATA\ExtOnnxVision\trt-engines\*" -Force
```

Make every layer forget its remembered model path:

```powershell
Remove-Item "$env:LOCALAPPDATA\ExtOnnxVision\layers\*" -Force
```

Both are safe: the first costs a recompile, the second means retyping paths.

---

## Building from source

Needs Visual Studio 2022 with the *Desktop development with C++* workload,
CMake 3.14 or newer, and git. No CUDA toolkit and no TensorRT installation are
needed to build.

```
git clone --recurse-submodules https://github.com/blackdown/pixera-onnx-vision.git
cd pixera-onnx-vision
cmake -B build
cmake --build build --config Release --target ExtOnnxVision
```

The extension is written to
`render-bridge-sdk/RenderBridge/RX/modules/ExtOnnxVision.dll`. To deploy
straight into a PIXERA installation instead, pass
`-DPIXERA_BUILD="C:/Program Files/AV Stumpfl/Pixera/<build>"` to the first
`cmake` command.

If you cloned without `--recurse-submodules`, run
`git submodule update --init` before configuring.

---

## Third-party components

- **RenderBridge SDK** — AV Stumpfl, included as a git submodule from
  [avstumpfl/render-bridge-sdk](https://github.com/avstumpfl/render-bridge-sdk)
  under its own licence.
- **TensorRT headers** — NVIDIA, Apache License 2.0, vendored in
  [`third_party/tensorrt-headers`](third_party/tensorrt-headers). TensorRT
  itself is not distributed here; it is installed separately under NVIDIA's
  terms.
- **Models** are not included. Check the licence of any model you use.
