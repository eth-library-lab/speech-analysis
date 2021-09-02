# eth-library-lab/speech

![AID Testing](https://github.com/eth-library-lab/speech/actions/workflows/aid-ci.yml/badge.svg)

> Speech Analysis Package

This project is based on the AID Project. Common topics are discussed in the [AID docs](https://aid.autoai.org). Make sure to read it!

## Usage

* Currently this model only accepts ```.wav``` files. 
* Once installed and deployed with AID, you can test it by ```curl -X POST -F file=@mlk.wav http://127.0.0.1:8080/infer```.
* Currently, the model only supports speech-to-text and then analyze on top of the text. More features will join soon.

## Reference

1. [DeepSpeech](https://github.com/mozilla/DeepSpeech)
