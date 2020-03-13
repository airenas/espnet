# Sabina Text-to-Speech recipe

Written by Airenas Vaičiūnas @ VMU, Kaunas (2020/03/13)

## tts1 recipe

`tts1` recipe is based on Tacotron2 [1] (spectrogram prediction network) w/o WaveNet.  
Tacotron2 generates log mel-filter bank from text and then converts it to linear spectrogram using inverse mel-basis.  
Finally, phase components are recovered with Griffin-Lim.
