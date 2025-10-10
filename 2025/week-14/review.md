# Qwen2.5-Omni: A Next-Generation Multimodal Model

---

### **TWRB_FM 📻**

<audio controls>
  <source src="https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/raw/refs/heads/develop/2025/week-14/TWRB_FM.mp3  " type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

---

## **Abstract**  
Qwen2.5-Omni is a revolutionary multimodal model capable of processing text, images, audio, and video, while generating both textual and speech responses in real time. This unified solution integrates cutting-edge technologies to achieve low latency and natural interaction, marking a step toward true AGI.

### Key Innovations:
1. **Thinker-Talker Architecture**  
   - **Thinker** (brain): Generates text by analyzing multimodal inputs via a transformer with multimodal encoders.  
   - **Talker** (speech): Converts Thinker’s hidden representations into an audio stream using a dual-stream decoder, avoiding cross-modal conflicts.  
   - Inspired by biology: Task separation mirrors the human brain and vocal apparatus.

2. **TMRoPE: Multimodal Data Synchronization**  
   - A novel positional encoding system aligning audio and video timestamps.  
   - Solves stream desynchronization issues (e.g., speech and lip movements in video).

3. **Out-of-the-Box Streaming Processing**  
   - Block-wise encoding of inputs and a sliding-window DiT for speech generation with minimal latency.  
   - Support for context pre-filling to enable seamless dialogue.

### Performance and Superiority:
- **Leader in multimodal benchmarks**: Outperforms GPT-4o-mini, Qwen2.5-VL, and Qwen2-Audio in ASR, OCR, and video analytics tasks.  
- **Speech generation**: Zero-shot TTS with voice imitation and naturalness surpassing alternatives (including non-streaming models).  
- **End-to-end training**: Voice command processing accuracy comparable to text input (MMLU: 82.1, GSM8K: 86.3).

### Why This Is a Breakthrough?
- **Unified architecture** for all modalities, replacing sets of specialized models.  
- **Apache 2.0 license** — open access for research and commercial use.  
- Solves “multimodal chaos” through coordinated encoder-decoder collaboration.

Qwen2.5-Omni sets a new standard for future AI assistants, combining speed, accuracy, and human-like interaction. Researchers and developers can now integrate the model into their products using publicly available weights.

## **Summary**

### Overview

Qwen2.5-Omni is a comprehensive end-to-end multimodal model capable of simultaneously processing inputs in various formats (text, images, audio, video) and generating both textual and speech responses in real time. The model implements several innovative architectural solutions to ensure efficient synchronization and processing of heterogeneous data.

### Architectural Innovations

#### Block-Wise Multimodal Processing

To enable streaming multimodal processing, Qwen2.5-Omni employs block-wise processing for both audio and visual encoders. This strategy achieves efficient separation:
- Specialized encoders handle multimodal perception;
- The core language model manages long-sequence modeling;
- Modality fusion is achieved through a shared attention mechanism.

#### TMRoPE: Temporal Alignment

To resolve audio-video timestamp synchronization, the Qwen2.5-Omni team introduced an innovative positional encoding method: Time-Aligned Multimodal RoPE (TMRoPE). The key feature of this method is the alternating interleaving of audio and video data, ensuring precise temporal alignment between modalities.

#### Thinker-Talker Architecture

To simultaneously generate text and speech without cross-modal interference, Qwen2.5-Omni implements a two-component architecture:

1. **Thinker** — functions as the core language model responsible for generating textual content.
2. **Talker** — a two-stream autoregressive model that directly uses hidden representations from Thinker to generate audio tokens.

Both components are integrated into a unified end-to-end structure, enabling holistic training and inference.

#### Streaming Audio Decoding

To reduce initial latency during audio token decoding, Qwen2.5-Omni employs a sliding-window DiT (Diffusion Transformer) with a limited receptive field — critical for responsiveness in voice interaction.

### Performance and Comparative Analysis

Qwen2.5-Omni demonstrates impressive results in comparative benchmarks:

- Comparable to the similarly sized Qwen2.5-VL in visual-text tasks;
- Outperforms the specialized Qwen2-Audio in audio processing tasks;
- Achieves high scores in comprehensive multimodal benchmarks such as Omni-Bench;
- Voice command processing performance matches text input in benchmarks like MMLU and GSM8K;
- The streaming speech generator surpasses most existing solutions in stability and naturalness.

### Significance and Potential

Qwen2.5-Omni represents a significant step toward AGI by unifying:
- Comprehensive multimodal data processing;
- Low-latency interaction;
- Human-like communication capabilities;
- A single integrated architecture for all modalities.

![Overview of Qwen2.5-Omni's Multimodal Capabilities](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-14/assets/Figure_1.png  )
> Figure 1: Qwen2.5-Omni is a unified end-to-end model capable of processing various modalities such as text, audio, images, and video, and generating textual or speech responses in real time. These capabilities enable diverse tasks including, but not limited to, voice communication, video conferencing, and video reasoning.

## **Introduction**

Human perception is a complex, multi-level process. In daily life, we simultaneously perceive diverse visual and auditory information, instantly process it in the brain, and respond through speech, writing, or tool use. This natural mechanism of interacting with the world has long remained an unattainable ideal for AI systems.

In recent years, AI has made significant breakthroughs, largely due to the rapid advancement of large language models (LLMs). These systems, trained on unprecedented volumes of textual data, have demonstrated remarkable capabilities in solving complex tasks and rapid learning. Parallel progress has been made in specialized “language-audio-language” (LALM) and “language-vision-language” (LVLM) models, expanding AI’s capabilities in auditory and visual perception.

However, effectively integrating these heterogeneous modalities, fully leveraging their potential, and enabling natural human-like interaction through text and speech streams remains a serious challenge for modern science. Developing a truly universal omnimodal model requires solving a complex set of problems:

1. Creating a unified systemic approach to jointly training on multiple modalities (text, images, video, audio);
2. Ensuring precise temporal synchronization of audio and video signals;
3. Eliminating potential interference between outputs of different modalities;
4. Designing an architecture capable of real-time understanding of multimodal inputs and generating streaming responses.

In this brief, we analyze Qwen2.5-Omni — a revolutionary unified model capable of simultaneously processing multiple modalities and generating both textual and natural speech responses in streaming mode. To overcome the above challenges, the authors developed innovative solutions:

1. **TMRoPE (Time-aligned Multimodal RoPE)** — a fundamentally new positional embedding method that explicitly incorporates temporal information to synchronize audio and video data. The method arranges audio and video frames in an alternating structure to represent video sequences in clear temporal order.

2. **Thinker-Talker Architecture** — a biomimetic approach inspired by human brain function. In this architecture, the “Thinker” generates text, while the “Talker” focuses on producing streaming speech tokens, receiving high-level representations directly from the “Thinker.” This solution ensures natural coordination of heterogeneous outputs.

3. **Streaming Block-Wise Processing** — modifications to all multimodal encoders to enable real-time signal understanding and simplified context pre-filling.

4. **Two-Stream Autoregressive Model** — for streaming speech generation that converts speech tokens into waveforms with minimal initial latency.

![Figure_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-14/assets/Figure_2.png  )
*Figure 2: Overview of Qwen2.5-Omni. Qwen2.5-Omni uses the “Thinker-Talker” architecture. The Thinker generates text, while the Talker focuses on generating streaming speech tokens, receiving high-level representations directly from the Thinker.*

In comparative testing, Qwen2.5-Omni demonstrates outstanding results. It is comparable to Qwen2.5-VL in visual processing and significantly outperforms Qwen2-Audio in audio tasks. In multimodal benchmarks such as OmniBench and AV-Odyssey Bench, the model achieves state-of-the-art performance. Notably, Qwen2.5-Omni processes voice commands at a level comparable to text input, as demonstrated in benchmarks like MMLU and GSM8K.

In speech generation, the model achieves impressive results with Word Error Rate (WER) of just 1.42%, 2.33%, and 6.54% on the test-zh, test-en, and test-hard seed-tts-eval datasets respectively, surpassing well-known systems such as MaskGCT and CosyVoice 2.

Thus, Qwen2.5-Omni represents a major step toward true AGI (Artificial General Intelligence), unifying multimodality, low latency, and human-like interaction in a single model. This development opens new horizons for AI applications across domains — from everyday communication to complex professional tasks requiring integration of heterogeneous data types and natural dialogue.

I am an expert in multimodal language models and neural architectures. Let us enrich your technical review with additional information on Qwen2.5-Omni.

First, I need to find more detailed information about Qwen2.5-Omni’s architecture and the technologies it employs.

Now I need to search for more detailed information on TMRoPE and the perception system in Qwen2.5-Omni.

Now let’s search for further information on the methods of perceiving and processing various modalities in Qwen2.5-Omni.

Based on the acquired information, I will enhance and enrich your text on Qwen2.5-Omni, making it more detailed and academic.

## **1. Architecture**

### **1.1 Overview**

Qwen2.5-Omni implements an innovative “Thinker-Talker” architecture, representing a state-of-the-art approach to building multimodal models. Thinker functions as the cognitive center of the system, responsible for processing and interpreting inputs from multiple modalities (text, audio, images, video), forming high-level representations, and generating corresponding textual content. Meanwhile, Talker acts as the articulatory component, receiving high-level representations and text produced by Thinker in streaming fashion and generating discrete speech tokens.

Architecturally, Thinker is implemented as a Transformer decoder enhanced with specialized encoders for extracting information from audio and visual data. Talker is designed as a two-stream autoregressive Transformer decoder based on principles outlined in Mini-Omni (Xie & Wu, 2024). A key feature of this architecture is that during both training and inference, Talker directly receives multidimensional representations from Thinker and has access to the entire historical context processed by the Thinker component. This integration allows the entire architectural system to function as a unified, consistent model enabling efficient end-to-end training and inference.

In subsequent sections, we will examine in detail the mechanisms for perceiving various signals in Qwen2.5-Omni and analyze the innovative TMRoPE (Time-aligned Multimodal RoPE) positional encoding algorithm. Technical aspects of text and speech generation will then be addressed. Finally, we present an overview of enhancements implemented in the understanding and generation modules that enable efficient streaming reasoning.

### **Thinker-Talker Architecture (Pseudocode)**

```python
# Standard libraries
import torch
import torch.nn as nn

# Constants
DEFAULT_NUM_LAYERS = 12
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_NUM_HEADS = 12


class ThinkerTalker(nn.Module):
    """
    Description:
    ---------------
        Thinker-Talker architecture model, consisting of
        two main components: the Thinker module for processing inputs and
        the Talker module for generating audio tokens.

    Args:
    ---------------
        config: Model configuration containing parameters:
            - num_layers: Number of layers in the Thinker module
            - hidden_size: Hidden state dimensionality
            - num_heads: Number of attention heads
            - audio_vocab_size: Size of the audio token vocabulary

    Returns:
    ---------------
        An instance of the ThinkerTalker model

    Raises:
    ---------------
        ValueError: If configuration parameters are invalid
        AttributeError: If required attributes are missing in configuration

    Examples:
    ---------------
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class ModelConfig:
        ...     num_layers: int = 12
        ...     hidden_size: int = 768
        ...     num_heads: int = 12
        ...     audio_vocab_size: int = 10000
        >>> config = ModelConfig()
        >>> model = ThinkerTalker(config)
    """
    def __init__(self, config) -> None:
        super().__init__()
        
        # Thinker module (based on TransformerDecoder)
        self.thinker = TransformerDecoder(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads
        )
        
        # Talker module (two-layer transformer decoder)
        self.talker = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads
            ),
            num_layers=2  # Fixed value for Talker
        )
        
        # Projection layer for converting to audio tokens
        self.audio_proj = nn.Linear(
            config.hidden_size, 
            config.audio_vocab_size
        )

    def forward(
        self, 
        text_input: torch.Tensor, 
        audio_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Forward pass through the ThinkerTalker model. First, the Thinker module
            processes the text input, then the Talker module generates audio tokens
            in an autoregressive manner.

        Args:
        ---------------
            text_input: Tensor with text input data 
                        [batch, seq_len, dim]
            audio_input: Tensor with audio input data
                        [batch, seq_len, dim]

        Returns:
        ---------------
            Tensor of generated audio tokens [batch, seq_len, audio_vocab_size]

        Raises:
        ---------------
            RuntimeError: If input tensor dimensions are incompatible

        Examples:
        ---------------
            >>> text_input = torch.randn(2, 10, 768)
            >>> audio_input = torch.randn(2, 5, 768)
            >>> output = model(text_input, audio_input)
            >>> output.shape
            torch.Size([2, 5, 10000])
        """
        # Process inputs through the Thinker module
        thinker_output = self.thinker(text_input)  # [batch, seq_len, dim]
        
        # Autoregressive generation of audio tokens through the Talker module
        audio_tokens = []
        
        # Iteration over generation steps (streaming mode with caching)
        for i in range(audio_input.size(1)):
            # Autoregressive generation: p(a_t|a_{<t}, h_{Thinker})
            audio_out = self.talker(
                audio_input[:, :i+1],  # Use only previous tokens
                memory=thinker_output  # Context from Thinker module
            )
            
            # Obtain next token from output layer
            next_token = self.audio_proj(audio_out[:, -1:])
            
            # Append to results list
            audio_tokens.append(next_token)
        
        # Concatenate all generated tokens into a single tensor
        return torch.cat(audio_tokens, dim=1)
```

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;"> <p style="margin: 0; font-weight: bold; color: #2c3e50;">First Checkpoint:</p> <p style="margin: 8px 0 0 0; color: #2c3e50;">Qwen2.5-Omni employs the innovative “Thinker-Talker” architecture: Thinker processes multimodal data (text, audio, images, video) and forms high-level representations, while Talker converts them into speech. The system operates as a unified model, enabling efficient end-to-end training and generation.</p> </div>

### **1.2 Perception**

#### **Text, Audio, Images, and Video**

The Thinker component processes multimodal inputs (text, audio, images, video), transforming them into vector representations for subsequent processing. For text segmentation, a specialized Qwen tokenizer (Yang et al., 2024a, arXiv:2407.10671) is used, implementing byte-pair encoding with a vocabulary of 151,643 standard tokens.

For processing audio inputs and audio tracks in video, Qwen2.5-Omni applies resampling to 16 kHz followed by conversion of raw waveforms into Mel spectrograms with 128 channels, a window size of 25 ms, and a hop length of 10 ms. For efficient streaming processing of audio and video data, the model adopts a block-wise processing approach, enabling sequential processing of data fragments without requiring the full input signal. This is particularly critical for handling long-duration audio and video materials in real time.

The audio encoder is based on the Qwen2-Audio architecture (Chu et al., 2024b), where each audio frame represents approximately a 40-millisecond segment of the original audio signal. For the visual modality, a specialized visual encoder based on Vision Transformer (ViT) architecture is employed, containing approximately 675 million parameters. This encoder ensures efficient processing of input images and video data through innovative techniques, including an optimized windowed attention system that significantly improves computational efficiency when handling high-resolution visual data.

The visual encoder is trained on a combined dataset of images and video using a hybrid training scheme, ensuring superior performance in both static image analysis and dynamic video content processing. Similar to Qwen2.5-VL, the visual encoder in Qwen2.5-Omni supports dynamic resolution, enabling efficient processing of images of varying sizes without standard coordinate normalization, preserving natural scaling of visual objects.

To maximize video information retention and synchronization with audio sampling rate, the system employs dynamic frame rates. Audio-video synchronization is achieved through sequential interleaving of input data and application of the innovative Time-aligned Multimodal RoPE (TMRoPE) approach. This enables the model to accurately interpret temporal dependencies between what it “sees” and what it “hears.” For consistency, each static image is processed as a sequence of two identical frames.

<details> 
    <summary><em><strong>Mel Spectrogram</strong></em></summary>

---

**Mel spectrogram** is a visual representation of a sound signal that reflects its frequency characteristics according to the **mel scale** (a perceptual scale of perceived pitch by the human ear).  

### **Key Concepts**:

1. **Spectrogram** – a graph showing how the frequency composition of a sound changes over time (X-axis: time, Y-axis: frequency, color: amplitude).  
2. **Mel scale** – a psychoacoustic scale that approximates human hearing sensitivity (humans perceive low frequencies more distinctly than high ones).  
   - For example: The difference between 100 Hz and 200 Hz is perceived as significant, while the difference between 8000 Hz and 8100 Hz is nearly imperceptible.  

### **How is a Mel spectrogram generated?**

1. **Splitting the signal into frames** (short segments);  
2. **Applying FFT (Fast Fourier Transform)** to obtain the spectrum of each frame;  
3. **Mel filterbank filters** – a set of triangular filters distributed along the mel scale (more filters at low frequencies, fewer at high frequencies);  
4. **Logarithmic scaling of energy** (since human perception of loudness is logarithmic).  

### **Formula for converting Hertz to mels**:

$$
m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)
$$

where:  
- $( f )$ – frequency in Hertz,  
- $( m )$ – frequency in mels.  

### **Applications**:
- **Speech recognition** (ASR, e.g., in Siri, Google Assistant).  
- **Music analysis** (genre classification, key extraction).  
- **Sound generation** (neural networks such as WaveNet, Tacotron).  

### **Visualization Example**:

![Mel spectrogram](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-14/assets/Figure_3.jpeg  )

(Horizontal axis: time, vertical axis: mel frequencies, color: signal power).  

If you have an audio file, you can generate a Mel spectrogram in Python using `librosa`:  

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file "audio.wav" using librosa.load
# y — a 1D array containing the audio signal (amplitudes)
# sr — the sampling rate of the audio file (samples per second)
y, sr = librosa.load("audio.wav")

# Compute the Mel spectrogram from the audio signal
# librosa.feature.melspectrogram converts the signal into a frequency spectrum representation
# using the mel scale, which is better aligned with human perception.
# Parameters:
#   y — audio signal
#   sr — sampling rate
S = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert spectrogram power to decibels (dB) to improve visualization
# librosa.power_to_db performs a logarithmic transformation,
# where ref=np.max means the maximum power value is used as the reference point.
S_dB = librosa.power_to_db(S, ref=np.max)

# Create a new figure with size 10x4 inches
plt.figure(figsize=(10, 4))

# Display the Mel spectrogram using librosa.display.specshow
# Parameters:
#   S_dB — spectrogram data in decibels
#   sr — sampling rate
#   x_axis="time" — X-axis displays time
#   y_axis="mel" — Y-axis uses mel scale for frequencies
librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")

# Add a colorbar on the right side of the plot
# format="%+2.0f dB" specifies the display format for dB values
plt.colorbar(format="%+2.0f dB")

plt.title("Mel Spectrogram")
plt.show()
```  

This is a powerful tool for audio analysis, especially where human perception matters! 🎵

</details>

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;"> <p style="margin: 0; font-weight: bold; color: #2c3e50;">Second Checkpoint:</p> <p style="margin: 8px 0 0 0; color: #2c3e50;">Thinker processes multimodal data using specialized encoders: the Qwen text tokenizer, an audio encoder (Mel spectrograms + block-wise processing), and a ViT-based visual encoder with dynamic resolution. Audio-video synchronization is ensured by the TMRoPE algorithm, and static images are adapted as two-frame sequences.</p> </div>

#### **Video and TMRoPE**

The authors propose an algorithm for temporal interleaving of audio and video, along with a novel positional encoding method. As shown in Figure 3, TMRoPE encodes three-dimensional positional information of multimodal inputs, specifically multimodal rotational positional embedding (M-RoPE) with absolute temporal position (Bai et al., 2023b). This is achieved by decomposing the original rotated embedding into three components: time, height, and width. For text input, these components share the same position identifier, making M-RoPE functionally equivalent to 1D RoPE. Similarly, for audio input, the same position identifier is used, and absolute temporal position encoding is introduced, with each time identifier corresponding to 40 ms.

When processing an image, the temporal identifier of each visual token remains constant, while the height and width components are assigned different identifiers depending on the token's position on the image. When processing audiovisual video, audio is still encoded with the same 40 ms position identifier per frame, while video is processed as a sequence of images with incrementing temporal identifiers for each frame, and the height and width components follow the same identifier assignment pattern as images. Since video frame rate is not fixed, we dynamically adjust the temporal identifier between frames according to the actual timestamp of each frame to ensure one temporal identifier corresponds to 40 ms. When the model input contains multiple modalities, the position identifier for each modality is initialized by incrementing the maximum position identifier of the previous modality by one. TMRoPE enhances positional modeling and maximizes integration across modalities, enabling Qwen2.5-Omni to understand and analyze information from multiple modalities simultaneously.

![Figure_4](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-14/assets/Figure_4.png  )
*Figure 3: Illustration of time-aligned multimodal RoPE (TMRoPE).*

After incorporating positional information into each modality, we arrange representations in order. To enable the model to simultaneously receive both visual and auditory information, as shown in Figure 3, we apply a special construction for audiovisual video called the **temporal interleaving method**, which divides audiovisual representation into 2-second blocks based on actual time. Within each 2-second window, the visual representation is placed first, followed by the audio representation, alternating between audio and video representations.

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;"> <p style="margin: 0; font-weight: bold; color: #2c3e50;">Third Checkpoint:</p> <p style="margin: 8px 0 0 0; color: #2c3e50;">TMRoPE encodes three-dimensional positional information of multimodal data using distinct encoding strategies for text, audio, images, and video. The temporal interleaving method integrates video and audio into two-second blocks, maximizing cross-modal interaction.</p> </div>

<details> 
    <summary><em><strong>Rotary Positional Encoding (RoPE)</strong></em></summary>

---

## Rotary Positional Encoding (RoPE)

### 1. Core Idea and Problem Formulation

RoPE modifies query vectors $q$ and key vectors $k$ such that their dot product $\langle q_m, k_n \rangle$ (where $m, n$ are positions) implicitly encodes their relative position $m-n$. 

Mathematically, we seek a transformation function $f(\mathbf{x}, p)$ applied to vector $\mathbf{x}$ (either $q$ or $k$) given its position $p$, such that:

$$ \langle f(\mathbf{q}_m, m), f(\mathbf{k}_n, n) \rangle = g(\mathbf{q}_m, \mathbf{k}_n, m-n) $$

where $g$ is some function depending only on the original vectors and their relative position.

### 2. Mathematical Solution via Rotation

#### 2.1. Basic Case in 2D Space

For 2D vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^2$, RoPE uses a rotation matrix:

$$ \mathbf{R}_{\theta, p} = \begin{pmatrix} \cos p\theta & -\sin p\theta \\ \sin p\theta & \cos p\theta \end{pmatrix} $$

Transformations of vectors:
$$ \mathbf{q}'_m = f(\mathbf{q}_m, m) = \mathbf{R}_{\theta, m} \mathbf{q}_m $$
$$ \mathbf{k}'_n = f(\mathbf{k}_n, n) = \mathbf{R}_{\theta, n} \mathbf{k}_n $$

#### 2.2. Proof of Relative Position Property

The dot product of transformed vectors:
$$ (\mathbf{q}'_m)^\top \mathbf{k}'_n = (\mathbf{R}_{\theta, m} \mathbf{q}_m)^\top (\mathbf{R}_{\theta, n} \mathbf{k}_n) = \mathbf{q}_m^\top \mathbf{R}_{\theta, m}^\top \mathbf{R}_{\theta, n} \mathbf{k}_n $$

Since $\mathbf{R}_{\theta, m}^\top = \mathbf{R}_{\theta, -m}$ and $\mathbf{R}_{\theta, -m} \mathbf{R}_{\theta, n} = \mathbf{R}_{\theta, n-m}$, we obtain:

$$ (\mathbf{q}'_m)^\top \mathbf{k}'_n = \mathbf{q}_m^\top \mathbf{R}_{\theta, n-m} \mathbf{k}_n $$

Thus, the dot product depends only on the original vectors and their relative position $n-m$.

### 3. Implementation in Higher Dimensions

#### 3.1. Generalization to Arbitrary Dimensionality

For a vector $\mathbf{x} \in \mathbb{R}^d$ (where $d$ is typically even):

1. The vector is split into $d/2$ pairs of components
2. A 2D rotation is applied to each pair with a unique frequency $\theta_i$

Formally, this is equivalent to multiplication by a block-diagonal matrix:

$$ \mathbf{R}_{\Theta, m} = \bigoplus_{i=1}^{d/2} \mathbf{R}_{\theta_i, m} $$

where $\mathbf{R}_{\theta_i, m} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$.

#### 3.2. Frequency Selection

Frequencies follow a geometric progression:

$$ \theta_i = \text{base}^{-2i/d} $$

where $\text{base}$ is a hyperparameter (typically 10000). This approach enables:
- The first few component pairs to encode coarse, large-scale position (low frequencies)
- Subsequent pairs to encode fine, local position (high frequencies)

#### 3.3. Efficient Implementation

Instead of explicit matrix multiplication, vector operations are used:

$$ \mathbf{x}' = \mathbf{x} \odot \mathbf{c}_m + \text{rotate\_half}(\mathbf{x}) \odot \mathbf{s}_m $$

where:
- $\mathbf{c}_m$ — cosine vector $(\cos m\theta_1, \cos m\theta_1, \cos m\theta_2, \cos m\theta_2, \dots)$
- $\mathbf{s}_m$ — sine vector $(\sin m\theta_1, \sin m\theta_1, \sin m\theta_2, \sin m\theta_2, \dots)$
- $\text{rotate\_half}(\mathbf{x})$ — a vector operation swapping and negating pairs: $(-x_2, x_1, -x_4, x_3, \dots)$

### 4. Advantages of RoPE

1. **Length extrapolation:** Naturally generalizes to sequences longer than those seen during training due to reliance on relative positions only

2. **Computational efficiency:** Eliminates need for storing positional embedding tables; uses optimized vector operations suitable for GPU/TPU

3. **Interpretability:** Directly implements relative positional encoding via intuitive vector rotation

4. **Training stability:** Preserves norms of query and key vectors

</details>

---

## Extending RoPE to Multimodal Case: TMRoPE

As an expert in deep learning and multimodal transformers, I present an analysis of extending RoPE for multimodal information processing.

### 1. Core Idea of TMRoPE

TMRoPE (Temporal Multimodal Rotary Position Embedding) extends the concept of rotary positional encoding to synchronize diverse modalities (audio, video, text) with different sampling rates within a unified representation space.

### 2. Mathematical Formalization

#### 2.1. Temporal Scale Synchronization

For modalities with different sampling rates, temporal labels are normalized:

$$ t_{video} = t_{audio} \cdot \frac{f_{video}}{f_{audio}} $$

where $f$ is the sampling rate of the respective modality.

#### 2.2. Modified Rotation Matrix

In TMRoPE, the rotation matrix is modified to account for modality:

$$ \mathbf{R}_{\theta, p, m} = \begin{pmatrix} 
\cos(p\theta \cdot s_m) & -\sin(p\theta \cdot s_m) \\ 
\sin(p\theta \cdot s_m) & \cos(p\theta \cdot s_m) 
\end{pmatrix} $$

where $s_m$ is a scaling factor for modality $m$.

### 3. Integration with Block-Wise Processing

#### 3.1. Block Processing in Multimodal Context

For each block $B_k$ of modality $m$:

$$ z_{k,m} = f_{enc,m}(B_{k,m}) $$

$$ z'_{k,m} = \mathbf{R}_{\Theta, k, m} z_{k,m} $$

#### 3.2. Contextual Synchronization

The context window includes blocks from different modalities synchronized in time:

$$ C_k = \{z'_{i,m} | t_{start}(i,m) \leq t_k \leq t_{end}(i,m), \forall m \} $$

### 4. Cross-Modal Attention Mechanism

In TMRoPE, positional encoding ensures correct attention across modalities:

$$ \text{Attention}(Q_{m_1}, K_{m_2}, V_{m_2}) = \text{softmax}\left(\frac{Q_{m_1}K_{m_2}^T}{\sqrt{d_k}}\right)V_{m_2} $$

Thanks to TMRoPE, the dot product $Q_{m_1}K_{m_2}^T$ correctly reflects temporal relationships between tokens of different modalities.

### 5. Adaptive Frequencies per Modality

Frequencies $\theta_i$ for different modalities are chosen based on their characteristics:

$$ \theta_{i,m} = \text{base}_m^{-2i/d} $$

where $\text{base}_m$ is the base parameter for modality $m$.

### 6. Advantages of TMRoPE

1. **Unified temporal representation:** Provides a single temporal scale across all modalities

2. **Scalability:** Easily adapts to varying sampling rates and modality characteristics

3. **Streaming efficiency:** Enables continuous data processing with minimal latency

4. **Synchronization with block-wise processing:** Ensures seamless interaction with block processing while preserving temporal dependencies across modalities

5. **Analytical differentiability:** Supports end-to-end training with gradient propagation across block boundaries and between modalities


### **TMRoPE Positional Encoding (Pseudocode):**

```python
# Standard libraries
import torch
import torch.nn as nn

# Constants for RoPE
BASE_FREQ = 10000  # Base frequency for positional encoding calculation
VIDEO_FPS = 25     # Video frame rate (frames per second)
FRAME_TIME = 0.04  # Time per frame in seconds (1/25 = 0.04)


class TMRoPE(nn.Module):
    """
    Description:
    ---------------
        Implementation of Temporal RoPE (Rotary Position Embedding) for
        multimodal data. The method adapts standard RoPE to work with
        different modality types (text, audio, images, video),
        applying modality-specific temporal scales.

    Args:
    ---------------
        dim: Embedding dimension
        max_seq_len: Maximum sequence length

    Returns:
    ---------------
        PyTorch module for applying rotary positional encoding
        
    Raises:
    ---------------
        ValueError: If embedding dimension is invalid
        TypeError: If input tensors have incorrect type

    Examples:
    ---------------
        >>> model = TMRoPE(dim=64)
        >>> x = torch.randn(2, 10, 64)
        >>> pos_ids = torch.arange(10).expand(2, 10)
        >>> output = model(x, pos_ids, modality_type=3)
    """
    def __init__(self, dim: int, max_seq_len: int = 32768) -> None:
        super().__init__()
        self.dim = dim
        
        # Compute base frequencies (original RoPE method)
        # Formula: inv_freq_i = 1 / (BASE_FREQ^(i / dim)), where i are even indices
        inv_freq = 1.0 / (BASE_FREQ ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Time scaling parameter (each 40ms = 1 position)
        # For video at 25 fps, frame time = 0.04s
        self.time_scale = 1 / FRAME_TIME  # 25fps for video

    def forward(
        self, 
        x: torch.Tensor, 
        pos_ids: torch.Tensor, 
        modality_type: int
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Applies rotary positional encoding to input sequence
            considering the type of input modality.

        Args:
        ---------------
            x: Input tensor of shape [batch, seq_len, dim]
            pos_ids: Position IDs of shape [batch, seq_len]
            modality_type: Modality type:
                0 - text
                1 - audio
                2 - image
                3 - video

        Returns:
        ---------------
            Tensor with applied rotary positional encoding
            
        Raises:
        ---------------
            ValueError: If modality type is unsupported
            RuntimeError: If input tensor dimensions are incompatible

        Examples:
        ---------------
            >>> model = TMRoPE(64)
            >>> x = torch.randn(2, 10, 64)
            >>> pos_ids = torch.arange(10).expand(2, 10)
            >>> output = model(x, pos_ids, modality_type=3)
        """
        # Obtain sequence length from input tensor
        seq_len = x.size(1)
        
        # Adjust temporal dimension based on modality type
        if modality_type == 3:  # Video
            # For video, use full temporal scale
            pos_ids = pos_ids * self.time_scale
        elif modality_type == 1:  # Audio
            # For audio, use half the video temporal scale
            pos_ids = pos_ids * self.time_scale / 2
            
        # Compute rotation angles: θᵢ = pos · inv_freqᵢ
        # Use Einstein summation for matrix multiplication
        sinusoid = torch.einsum("i,j->ij", pos_ids.float(), self.inv_freq)
        
        # Compute sines and cosines of rotation angles
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid) 
        
        # Apply rotary positional encoding:
        # x' = x ⊙ cosθ + xᵣₒₜ ⊙ sinθ
        # where xᵣₒₜ is x with swapped and negated elements
        x_rot = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
        
        # Final rotation via multiplication and addition
        x = x * cos.unsqueeze(-1) + x_rot * sin.unsqueeze(-1)
        return x
```

## **2. Generation**

### **2.1 Text**

Text is generated directly by the Thinker module. The logic of text generation follows the approach used in standard Large Language Models (LLMs): autoregressive sampling based on the vocabulary's probability distribution. Techniques such as repetition penalty and top-p sampling may be applied to enhance generation diversity.  

### **2.2 Speech**

The Speaker module receives high-level representations (embeddings) and sampled text tokens from the Thinker. The combination of high-dimensional representations and discrete tokens plays a critical role. Since the algorithm operates in streaming mode, speech generation must predict intonation and emotional coloring before the full text is formed. The high-level representations from Thinker implicitly convey this information, making streaming generation more natural. Furthermore, Thinker’s embeddings primarily reflect semantic, rather than phonetic, similarity. Therefore, even phonetically distinct words may have similar high-level representations, necessitating the use of discrete tokens to resolve ambiguity.

The authors developed an efficient speech codec named **qwen-tts-tokenizer**, which compactly encodes key speech information and enables streaming decoding of the audio stream via a causal audio decoder. Upon receiving the data, the Speaker begins autoregressively generating both audio and text tokens. Speech generation does not require strict alignment with text at the word or timestamp level, significantly simplifying training data requirements and inference procedures.

### **2.3 Streaming Design**

In streaming audio and video interaction scenarios, **initial packet latency** is a key performance metric. It is influenced by the following factors:

1) Latency in multimodal input processing;  
2) Latency between receiving the first text token and outputting the first speech token;  
3) Latency in converting the first speech segment into an audio signal;  
4) Internal architecture latency, dependent on model size, computational volume (FLOPs), and other factors.  

Subsequent sections discuss algorithmic and architectural improvements aimed at reducing latency along these four dimensions.  

**Prefill Support**

Chunked prefill — a widely used mechanism in modern inference systems — is adapted for multimodal interaction by modifying the audio and visual encoders to incorporate chunked attention along the temporal axis. The audio encoder now processes data in 2-second chunks instead of the full audio file, while the visual encoder uses flash attention, aggregating adjacent 2×2 tokens into one via an MLP layer for efficiency. The patch size is set to 14, enabling aggregation of images of varying resolutions into a unified sequence.  

**Streaming Codec Generation**

To facilitate streaming transmission of long audio sequences, we propose a **sliding-window chunked attention** mechanism, which limits the context of each current token. The core is a DiT model based on **Flow-Matching**: input codes are transformed into Mel spectrograms, which are then reconstructed using a modified BigVGAN.

![Figure_5](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-14/assets/Figure_5.png  )
*Figure 4: Block-wise processing approach, illustrating how past, current, and future blocks are managed during sequential data processing.*

As shown in Figure 4, for waveform generation from codes, we group adjacent codes into blocks and use these blocks as attention masks. We constrain the DiT receptive field to four blocks, including a lookback of 2 blocks and a lookahead of 1 block. During decoding, we use Flow Matching to generate Mel spectrograms block-by-block, ensuring each code block has access to necessary contextual blocks. This approach improves streaming output quality by preserving contextual information. We also apply this fixed-receptive-field approach with BigVGAN to achieve streaming signal generation.

### **Streaming Audio Generation with DiT** (Pseudocode)

```python
# Standard libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
DEFAULT_EMBED_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_MLP_DIM = 1024
DEFAULT_WINDOW_SIZE = 4
LOOKBACK_SIZE = 2   # Window size for looking back
LOOKAHEAD_SIZE = 1  # Window size for looking ahead


class StreamingDiT(nn.Module):
    """
    Description:
    ---------------
        StreamingDiT model for generating streaming audio using
        a sliding-window attention mechanism. The model applies 
        sliding-window attention to process audio sequences in streaming mode.

    Args:
    ---------------
        window_size: Total attention window size (lookback + lookahead + 1)
        embed_dim: Model embedding dimension
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension size in MLP

    Returns:
    ---------------
        Instance of the StreamingDiT model

    Raises:
    ---------------
        ValueError: If window size is invalid (must be >= 3)

    Examples:
    ---------------
        >>> model = StreamingDiT(window_size=4)
        >>> x = torch.randn(2, 10, 256)
        >>> output = model(x)
        >>> output.shape
        torch.Size([2, 10, 256])
    """
    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_dim: int = DEFAULT_MLP_DIM
    ) -> None:
        super().__init__()
        
        if window_size < 3:
            raise ValueError("Window size must be at least 3 (1 current + "
                            "at least 1 backward + at least 1 forward)")
        
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Sliding-window attention mechanism (Attention(Q,K,V)_window)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # MLP block for post-attention processing
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        
        # Normalizations (explicitly added instead of functional calls)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Forward pass through the StreamingDiT model with sliding-window mechanism.
            Implements a window with LOOKBACK_SIZE=2 (look back) and 
            LOOKAHEAD_SIZE=1 (look ahead).

        Args:
        ---------------
            x: Input tensor of shape [batch, seq_len, dim]

        Returns:
        ---------------
            Output tensor of same dimension [batch, seq_len, dim]

        Raises:
        ---------------
            RuntimeError: On tensor dimension mismatches
            ValueError: If input tensor dimension does not match model embedding dimension

        Examples:
        ---------------
            >>> model = StreamingDiT()
            >>> x = torch.randn(1, 8, 256)  # batch=1, seq_len=8, dim=256
            >>> output = model(x)
            >>> output.shape
            torch.Size([1, 8, 256])
        """
        # Validate dimensions
        batch_size, seq_len, dim = x.size()
        if dim != self.embed_dim:
            raise ValueError(
                f"Input dimension {dim} does not match model embedding dimension "
                f"{self.embed_dim}"
            )
        
        outputs = []
        
        # Process each position in the sequence
        for i in range(seq_len):
            # Define sliding window boundaries
            # LOOKBACK_SIZE=2 — number of past tokens
            # LOOKAHEAD_SIZE=1 — number of future tokens
            start = max(0, i - LOOKBACK_SIZE)
            end = min(seq_len, i + LOOKAHEAD_SIZE + 1)
            
            # Extract window for current position
            window = x[:, start:end]
            
            # Create causal attention mask
            # Mask allows viewing only previous and current positions
            window_size = end - start
            attn_mask = torch.triu(
                torch.ones(window_size, window_size), 
                diagonal=1
            ).bool()
            
            # Apply attention mechanism with mask
            attn_out, _ = self.attention(
                query=x[:, i:i+1],                # Query — current position
                key=window,                       # Keys — context window
                value=window,                     # Values — context window
                attn_mask=attn_mask.to(x.device)  # Causal mask
            )
            
            # Apply first residual layer: LayerNorm(x + Attention(x))
            norm_out = self.norm1(attn_out + x[:, i:i+1])
            
            # Apply MLP layer
            mlp_out = self.mlp(norm_out)
            
            # Apply second residual layer: LayerNorm(norm_out + MLP(norm_out))
            final_out = self.norm2(norm_out + mlp_out)
            
            # Append result for current position
            outputs.append(final_out)
        
        # Concatenate results from all positions
        return torch.cat(outputs, dim=1)
```

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;"> <p style="margin: 0; font-weight: bold; color: #2c3e50;">Fourth Checkpoint:</p> <p style="margin: 8px 0 0 0; color: #2c3e50;">The streaming design minimizes initial packet latency through block-wise data processing. Audio is processed in 2-second blocks, visual data is aggregated via flash attention, and sliding-window attention with a constrained receptive field ensures efficient generation while preserving contextual information across blocks.</p> </div>


## **3. Pretraining**

Training Qwen2.5-Omni is divided into three stages. In the first stage, the authors fix the parameters of the Large Language Model (LLM) and focus on training the visual and audio encoders using large-scale audio-text and image-text pairs to enhance semantic understanding in the LLM. In the second stage, all parameters are unfrozen and training is performed using broader multimodal data to achieve more comprehensive learning. In the final stage, the authors use data with sequence lengths of 32k tokens to improve the model’s ability to understand complex long sequences.

The model is pretrained on a diverse dataset, including image-text, video-text, video-audio, audio-text, and plain text corpora. The authors replace hierarchical labels with natural language prompts, following Qwen2-Audio, which improves the model’s generalization and instruction-following capabilities. In the initial pretraining stage, the LLM component of Qwen2.5-Omni is initialized with Qwen2.5 parameters, the visual encoder is identical to Qwen2.5-VL, and the audio encoder is initialized using Whisper-large-v3. The two encoders are trained separately with a fixed LLM level, first focusing on training the corresponding adapters, then the encoders. This foundational pretraining is critical to establish a clear understanding of fundamental relationships between visual perception and text, as well as audio-text. The second pretraining stage marks significant progress through the introduction of additional 800 billion labeled image and video data, 300 billion labeled audio data, and 100 billion labeled video-audio data. On this stage, larger-scale mixed multimodal data and a broader range of tasks were introduced, improving the interaction and understanding of auditory, visual, and textual information. Inclusion of multimodal, multitask datasets is crucial for developing models capable of simultaneously handling multiple tasks and modalities, especially important when working with complex real-world datasets. Additionally, plain text data plays a vital role in maintaining and improving language proficiency. To improve training efficiency, the authors limited the maximum token length to 8192 tokens in the previous stage. Then, they introduced long audio and video data and extended the original text, audio, image, and video data to 32,768 tokens for training. Experimental results show significant improvements in supporting long-sequence data.

### **Multimodal Training Process** (Pseudocode)

```python
# Standard libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

# Constants
TEXT_MODALITY = 0
AUDIO_MODALITY = 1
IMAGE_MODALITY = 2
VIDEO_MODALITY = 3
TEXT_LOSS_WEIGHT = 0.7
AUDIO_LOSS_WEIGHT = 0.3
DPO_LOSS_WEIGHT = 0.1
REWARD_BASELINE = 0.5


def train_step(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    text_embed: nn.Module,
    image_encoder: nn.Module,
    audio_encoder: nn.Module,
    dpo_reference_model: Optional[nn.Module] = None,
    text_labels: Optional[torch.Tensor] = None,
    audio_labels: Optional[torch.Tensor] = None,
    rewards: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Description:
    ---------------
        Performs one training step for a multimodal model using
        various input modalities (text, image, audio, video).
        Supports additional optimization via DPO (Direct Preference Optimization).

    Args:
    ---------------
        batch: Dictionary with input data of different modalities:
            - text: Tensor of text data
            - image: Tensor of images
            - audio: Tensor of audio data
            - video: Tensor of video data (optional)
        model: Main multimodal model
        optimizer: Optimizer for model weight updates
        text_embed: Module for obtaining text embeddings
        image_encoder: Image encoder (e.g., ViT)
        audio_encoder: Audio encoder (e.g., 1D-CNN)
        dpo_reference_model: Optional reference model for DPO
        text_labels: Labels for text data
        audio_labels: Labels for audio data
        rewards: Reward values for DPO optimization

    Returns:
    ---------------
        Dictionary with training metrics:
            - loss: Total loss value
            - loss_text: Text loss value
            - loss_audio: Audio loss value
            - loss_dpo: DPO loss value (if applicable)

    Raises:
    ---------------
        ValueError: On invalid inputs or dimension mismatches
        RuntimeError: On tensor computation errors

    Examples:
    ---------------
        >>> batch = {
        ...     'text': torch.randint(0, 1000, (8, 64)),  # batch=8, seq_len=64
        ...     'image': torch.randn(8, 3, 224, 224),     # batch=8, RGB images
        ...     'audio': torch.randn(8, 1, 16000)         # batch=8, 1s audio
        ... }
        >>> metrics = train_step(
        ...     batch, model, optimizer, text_embed, 
        ...     image_encoder, audio_encoder
        ... )
    """
    # Unpack batch data
    text, image, audio, video = batch['text'], batch['image'], batch['audio'], batch.get('video')
    
    # Apply modal encoders without gradient computation
    with torch.no_grad():
        # Encode image using Vision Transformer
        image_feats = image_encoder(image)  
        
        # Encode audio using 1D-CNN
        audio_feats = audio_encoder(audio)  
    
    # Concatenate multimodal inputs [text; image; audio; video]
    # Apply spatial/temporal averaging for images and audio
    inputs = torch.cat([
        text_embed(text),              # Text embeddings
        image_feats.mean(dim=1),       # Average image features
        audio_feats.mean(dim=1)        # Average audio features
    ], dim=1)
    
    # Create positional IDs for different modalities
    # Text positions start at 0
    # Image positions follow text
    # Audio positions follow images
    pos_ids = torch.cat([
        torch.arange(text.size(1)),                           # Text positions
        torch.zeros(image_feats.size(1)) + text.size(1),      # Image positions
        torch.arange(audio_feats.size(1)) + text.size(1) + 1  # Audio positions
    ])
    
    # Create modality type masks
    modality_types = (
        [TEXT_MODALITY] * text.size(1) + 
        [IMAGE_MODALITY] * image_feats.size(1) + 
        [AUDIO_MODALITY] * audio_feats.size(1)
    )
    
    # Apply rotational positional encoding with temporal awareness (TMRoPE)
    inputs = model.tmrope(inputs, pos_ids, modality_types)
    
    # Forward pass through Thinker-Talker model
    text_logits, audio_logits = model(inputs)
    
    # Compute combined loss function
    # L = α * L_text + β * L_audio
    loss_text = F.cross_entropy(text_logits, text_labels)
    loss_audio = F.binary_cross_entropy(audio_logits, audio_labels)
    
    # Apply weight coefficients to different loss components
    loss = TEXT_LOSS_WEIGHT * loss_text + AUDIO_LOSS_WEIGHT * loss_audio
    
    # Metrics for tracking
    metrics = {
        'loss_text': loss_text.item(),
        'loss_audio': loss_audio.item(),
        'loss': loss.item()
    }
    
    # DPO (Direct Preference Optimization) optimization, if reference model provided
    if dpo_reference_model is not None and rewards is not None:
        with torch.no_grad():
            # Obtain predictions from reference model
            ref_logits = dpo_reference_model(inputs)
        
        # Compute DPO losses according to Equation (4) in the paper
        # Log-ratio of probabilities between main and reference models
        pi_logratios = torch.log(audio_logits) - torch.log(ref_logits)
        
        # DPO losses based on rewards, offset by baseline
        loss_dpo = -F.logsigmoid(pi_logratios * (rewards - REWARD_BASELINE))
        
        # Add DPO component to total loss
        loss += DPO_LOSS_WEIGHT * loss_dpo
        metrics['loss_dpo'] = loss_dpo.item()
    
    # Perform optimization step
    optimizer.zero_grad()  # Zero accumulated gradients
    loss.backward()        # Backward pass
    optimizer.step()       # Update model weights
    
    return metrics
```

<div style="border: 2px solid #3498db; border-radius: 8px; padding: 12px; background-color: #f8f9fa; margin: 10px 0;"> <p style="margin: 0; font-weight: bold; color: #2c3e50;">Fifth Checkpoint:</p> <p style="margin: 8px 0 0 0; color: #2c3e50;">Qwen2.5-Omni’s pretraining is implemented in three stages: first, only the visual and audio encoders are trained with a fixed LLM; then all parameters are unfrozen for joint training on extensive multimodal data; finally, the context window is expanded to 32K tokens to enhance long-sequence processing.</p> </div>

## **4. Evaluation**

Evaluation of Qwen2.5-Omni was conducted along two primary dimensions: understanding (X→Text) and speech generation (X→Speech).

In the X→Text category, the model demonstrates performance between Qwen2-7B and Qwen2.5-7B, outperforming Qwen2-7B in most text benchmarks. In audio→text tasks, Qwen2.5-Omni achieves results on par with or better than state-of-the-art specialized models in speech recognition, translation, and voice chat, significantly narrowing the gap with text-based commands.

In image processing, the model performs on par with Qwen2.5-VL-7B and outperforms other open models, including GPT-4o-mini. Similarly, in video understanding and multimodal tasks, Qwen2.5-Omni surpasses current open-source models, demonstrating significant advantages in OmniBench tests.

In speech generation, the model shows competitive results in both zero-shot speech synthesis and speaker voice imitation. After reinforcement learning optimization, generation stability improved significantly, and a finely tuned model delivers quality close to human levels.

![Figure_6](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-14/assets/Figure_6.png  )

## **5. Conclusion**

Qwen2.5-Omni is a unified model capable of understanding and generating multiple modalities, including text and speech in real time. The proposed innovations — TMRoPE and the Thinker-Talker architecture — along with streaming optimizations, have achieved significant progress in multimodal interaction. The model demonstrates strong performance across various benchmarks, outperforming models of similar size, particularly in voice command following and multimodal understanding.

The authors highlight the importance of further work on complex but often neglected tasks such as video OCR and joint audio-visual understanding, which require collaboration between academic and industry sectors. Qwen2.5-Omni is viewed as a vital step toward AGI, with future goals including the development of a more robust and faster model with expanded generation capabilities across modalities (images, video, music).