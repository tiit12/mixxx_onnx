# Mixxx (phrase-analysis) kompileerimine Ubuntus

Juhend Ubuntu / Ubuntu Studio 26.04 või uuema jaoks. Kompileerib Mixxxi koos
neurovõrgul põhineva downbeat'i ja fraasianalüüsiga (ONNX Runtime).

## 1. Sõltuvused

```sh
sudo apt update
sudo apt install git libonnxruntime-dev
```

`libonnxruntime-dev` on `universe` repos, mis on Ubuntu Studios vaikimisi
sees. Kui paketti ei leita: `sudo add-apt-repository universe`.

## 2. Lähtekood

```sh
git clone https://github.com/tiit12/mixxx_onnx.git mixxx
cd mixxx
git checkout phrase-analysis
git remote add upstream https://github.com/mixxxdj/mixxx.git
```

Kloonimine laeb alla kogu Mixxxi ajaloo (~700 MB).

## 3. Mixxxi build-sõltuvused

```sh
tools/debian_buildenv.sh setup
```

Skript paigaldab kompilaatori, CMake'i, Qt6 ja kõik teegid (küsib `sudo`
parooli).

## 4. Kompileerimine

```sh
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DOPTIMIZE=native \
    -DONNXRUNTIME=ON
cmake --build build --parallel 4
```

- `OPTIMIZE=native` optimeerib just selle masina protsessori jaoks. Kui build
  peab töötama ka teistes arvutites, kasuta `portable`.
- `--parallel 4`: 8 GB mälu juures ära kasuta rohkem lõimi, muidu võib mälu
  otsa saada. Kui kompilaator katkeb veaga `Killed`, vähenda arvu 2 peale.
- Esimene build võtab i5 8. põlvkonna masinal umbes 30–60 minutit, järgmised
  ainult muudetud failid.

## 5. Kontroll

```sh
build/mixxx-test --gtest_filter='Downbeat*:PhraseAnalyzer*:BeatsTest.*'
```

Kõik testid peavad läbima (`[  PASSED  ]`).

## 6. Käivitamine

Esimesel korral eraldi profiiliga, et olemasolev teek jääks puutumata:

```sh
build/mixxx --settings-path ~/mixxx-test-settings
```

**Tähelepanu:** ilma `--settings-path`'ita kasutab Mixxx kausta `~/.mixxx`,
mida jagab ka Ubuntu paketist paigaldatud Mixxx. Uuem versioon võib teegi
andmebaasi uuendada nii, et vana Mixxx seda enam ei ava. Tee enne varukoopia:

```sh
cp -a ~/.mixxx ~/.mixxx-varukoopia
```

Eelistustes tasub valida **Waveforms → RGB (OpenGL)**. Ubuntu Studio
PipeWire'iga sobib **Sound Hardware → JACK Audio Connection Kit**.

## 7. DJ-kontrollerid (valikuline)

USB/HID kontrollerite jaoks on vaja udev reegleid (ka pärast sammu 8, sest
`/usr/local` alla paigaldades need aktiivsesse kohta ei satu):

```sh
sudo cp res/linux/mixxx-usb-uaccess.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Seejärel ühenda kontroller uuesti.

## 8. Paigaldamine (valikuline)

```sh
sudo cmake --install build
```

Paigaldab Mixxxi koos neurovõrgu mudeliga kausta `/usr/local`.
`/usr/local/bin/mixxx` on PATH-is eespool kui paketi `/usr/bin/mixxx`.

## Uuendamine

```sh
git pull
cmake --build build --parallel 4
```

Mixxxi enda uuenduste sisse toomiseks: `git fetch upstream && git merge upstream/main`.

## Puhas build nullist

```sh
rm -rf build
```

ja seejärel uuesti samm 4.

## Probleemid

- **`Could NOT find OnnxRuntime`**: pakett puudub. Laadi
  [ONNX Runtime'i väljalase](https://github.com/microsoft/onnxruntime/releases)
  (`onnxruntime-linux-x64-<versioon>.tgz`), paki lahti ja lisa sammu 4
  käsule `-DONNXRUNTIME_ROOT=/tee/onnxruntime-linux-x64-<versioon>`.
- **Fraasiribasid pole näha**: lugu analüüsitakse laadimisel taustal, see
  võtab loo kohta umbes 15–30 sekundit. Logis
  (`~/mixxx-test-settings/mixxx.log`) on read `AnalyzerBars` ja
  `PhraseAnalyzer`.

Rohkem infot analüüsi, juhtobjektide ja tuunimise kohta:
[tools/beatthis/README.md](tools/beatthis/README.md).
