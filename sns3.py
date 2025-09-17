import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import io
import tempfile
import pandas as pd
from audio_recorder_streamlit import audio_recorder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="Signal Simulator with Voice Processing",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .parameter-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class SignalSimulator:
    def __init__(self):
        self.sampling_rate = 44100
        if 'signal_history' not in st.session_state:
            st.session_state.signal_history = []
        if 'current_samples' not in st.session_state:
            st.session_state.current_samples = np.array([])
        if 'voice_samples' not in st.session_state:
            st.session_state.voice_samples = np.array([])
        if 'processed_voice' not in st.session_state:
            st.session_state.processed_voice = np.array([])

    def generate_signal(self, signal_type, params):
        """Generate different types of signals based on parameters"""
        length = int(params.get('length', 1000))
        n = np.arange(length)

        if signal_type == 'impulse':
            samples = np.zeros(length)
            pos = int(np.clip(params.get('position', 0), 0, length - 1))
            samples[pos] = params.get('amplitude', 1.0)

        elif signal_type == 'step':
            samples = np.zeros(length)
            pos = int(np.clip(params.get('position', 0), 0, length - 1))
            samples[pos:] = params.get('amplitude', 1.0)

        elif signal_type == 'sine':
            freq = params.get('frequency', 1.0)
            amp = params.get('amplitude', 1.0)
            phase = params.get('phase', 0.0)
            samples = amp * np.sin(2 * np.pi * freq * n / self.sampling_rate + phase)

        elif signal_type == 'exponential':
            amp = params.get('amplitude', 1.0)
            decay = params.get('decay', 0.1)
            samples = amp * np.exp(-decay * n)

        elif signal_type == 'chirp':
            f0 = params.get('f0', 1.0)
            f1 = params.get('f1', 10.0)
            t1 = params.get('t1', length / self.sampling_rate)
            samples = signal.chirp(n / self.sampling_rate, f0, t1, f1) * params.get('amplitude', 1.0)

        elif signal_type == 'noise':
            noise_type = params.get('noise_type', 'white')
            amp = params.get('amplitude', 1.0)
            if noise_type == 'white':
                samples = amp * np.random.randn(length)
            elif noise_type == 'pink':
                # Generate pink noise
                white = np.random.randn(length)
                f = np.fft.fftfreq(length)
                f = np.abs(f)
                f[0] = 1e-6  # Avoid division by zero
                pink_filter = 1 / np.sqrt(f)
                pink_filter[0] = 0
                samples = amp * np.fft.ifft(np.fft.fft(white) * pink_filter).real
        else:
            samples = np.zeros(length)

        return samples

    def apply_operations(self, samples, operations):
        """Apply signal operations"""
        processed = samples.copy()
        
        # Amplitude scaling
        processed = processed * operations.get('amplitude_scale', 1.0)
        
        # Amplitude shift
        processed = processed + operations.get('amplitude_shift', 0.0)
        
        # Time scaling
        time_scale = operations.get('time_scale', 1.0)
        if time_scale != 1.0:
            length = len(processed)
            x = np.arange(length)
            new_x = x * time_scale
            processed = np.interp(x, new_x, processed, left=0, right=0)
        
        # Time shift
        time_shift = int(operations.get('time_shift', 0))
        if time_shift != 0:
            processed = np.roll(processed, time_shift)
            
        return processed

    def plot_signal_plotly(self, samples, title="Signal", color="blue"):
        """Create interactive plot using Plotly"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=samples,
            mode='lines',
            name=title,
            line=dict(color=color, width=2)
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=400
        )
        return fig

    def plot_comparison_plotly(self, original, processed):
        """Create comparison plot using Plotly"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=original,
            mode='lines',
            name='Original',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=processed,
            mode='lines',
            name='Processed',
            line=dict(color='red', width=2),
            opacity=0.8
        ))
        fig.update_layout(
            title="Signal Comparison",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=400
        )
        return fig

def main():
    simulator = SignalSimulator()
    
    # App title
    st.markdown('<h1 class="main-header">🎵 Signal Simulator with Voice Processing</h1>', 
                unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Signal Generator", "🎤 Voice Signal Processing"])
    
    # ===== SIGNAL GENERATOR TAB =====
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="section-header">Original Signal Selection</div>', 
                       unsafe_allow_html=True)
            
            # Signal type selection
            signal_type = st.selectbox(
                "Signal Type",
                ['impulse', 'step', 'sine', 'exponential', 'chirp', 'noise'],
                index=2,
                help="Select the type of signal to generate"
            )
            
            # Common parameters
            st.markdown("**Common Parameters**")
            amplitude = st.slider('Amplitude', 0.1, 10.0, 1.0, 0.1)
            length = st.slider('Length (samples)', 100, 10000, 1000, 100)
            
            # Signal-specific parameters
            params = {'amplitude': amplitude, 'length': length}
            
            st.markdown("**Signal-Specific Parameters**")
            if signal_type in ['impulse', 'step']:
                params['position'] = st.slider('Position', 0, length-1, 0)
                
            elif signal_type == 'sine':
                params['frequency'] = st.slider('Frequency (Hz)', 0.1, 100.0, 1.0, 0.1)
                params['phase'] = st.slider('Phase (radians)', -np.pi, np.pi, 0.0, 0.1)
                
            elif signal_type == 'exponential':
                params['decay'] = st.slider('Decay Rate', 0.001, 1.0, 0.1, 0.001)
                
            elif signal_type == 'chirp':
                params['f0'] = st.slider('Start Frequency (Hz)', 0.1, 100.0, 1.0, 0.1)
                params['f1'] = st.slider('End Frequency (Hz)', 0.1, 100.0, 10.0, 0.1)
                params['t1'] = st.slider('Duration (s)', 0.1, 10.0, 1.0, 0.1)
                
            elif signal_type == 'noise':
                params['noise_type'] = st.selectbox('Noise Type', ['white', 'pink'])
            
            # Generate signal button
            if st.button("🔄 Generate Signal", type="primary"):
                with st.spinner('Generating signal...'):
                    samples = simulator.generate_signal(signal_type, params)
                    st.session_state.current_samples = samples
                    st.session_state.signal_history.append({
                        'type': signal_type,
                        'params': params.copy(),
                        'samples': samples.copy()
                    })
                st.success(f"Generated {signal_type} signal with {len(samples)} samples")
            
            # Display signal statistics
            if len(st.session_state.current_samples) > 0:
                samples = st.session_state.current_samples
                st.markdown("**Signal Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Length', 'Min Value', 'Max Value', 'Mean', 'Std Dev', 'RMS'],
                    'Value': [
                        len(samples),
                        f"{np.min(samples):.4f}",
                        f"{np.max(samples):.4f}",
                        f"{np.mean(samples):.4f}",
                        f"{np.std(samples):.4f}",
                        f"{np.sqrt(np.mean(samples**2)):.4f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Display original signal
            if len(st.session_state.current_samples) > 0:
                samples = st.session_state.current_samples
                fig = simulator.plot_signal_plotly(samples, f"Original {signal_type.title()} Signal")
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal Operations
                st.markdown('<div class="section-header">Signal Operations</div>', 
                           unsafe_allow_html=True)
                
                # Operation parameters
                with st.expander("🔧 Operation Controls", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        amplitude_scale = st.slider('Amplitude Scaling', 0.1, 5.0, 1.0, 0.1)
                        amplitude_shift = st.slider('Amplitude Shift', -5.0, 5.0, 0.0, 0.1)
                    
                    with col_b:
                        time_scale = st.slider('Time Scaling', 0.5, 2.0, 1.0, 0.1)
                        time_shift = st.slider('Time Shift (samples)', -100, 100, 0, 1)
                
                # Apply operations
                operations = {
                    'amplitude_scale': amplitude_scale,
                    'amplitude_shift': amplitude_shift,
                    'time_scale': time_scale,
                    'time_shift': time_shift
                }
                
                processed_samples = simulator.apply_operations(samples, operations)
                
                # Display comparison plot
                fig_comp = simulator.plot_comparison_plotly(samples, processed_samples)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Download processed signal
                if st.button("💾 Download Processed Signal"):
                    df = pd.DataFrame({
                        'Sample_Index': range(len(processed_samples)),
                        'Original': samples,
                        'Processed': processed_samples
                    })
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"processed_{signal_type}_signal.csv",
                        mime="text/csv"
                    )
            else:
                st.info("👈 Generate a signal to see plots and operations")
    
    # ===== VOICE PROCESSING TAB =====
    with tab2:
        st.markdown('<div class="section-header">Voice Signal Processing</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Voice Input**")
            
            # Method selection
            input_method = st.radio(
                "Choose input method:",
                ["Record Audio", "Upload File"],
                horizontal=True
            )
            
            if input_method == "Record Audio":
                st.info("🎤 Click the microphone button below to record audio")
                
                # Audio recorder
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#e8b62c",
                    neutral_color="#6aa36f",
                    icon_name="microphone",
                    icon_size="2x",
                )
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Process audio bytes
                    try:
                        # Save to temporary file and read
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_file.flush()
                            
                            # Read audio data
                            voice_data, sr = sf.read(tmp_file.name)
                            if voice_data.ndim > 1:
                                voice_data = voice_data[:, 0]  # Convert to mono
                            
                            st.session_state.voice_samples = voice_data
                            st.success(f"Recorded {len(voice_data)/sr:.2f} seconds of audio")
                            
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
            
            else:  # Upload File
                uploaded_file = st.file_uploader(
                    "Upload audio file",
                    type=['wav', 'mp3', 'ogg', 'flac'],
                    help="Upload an audio file to process"
                )
                
                if uploaded_file:
                    try:
                        voice_data, sr = sf.read(uploaded_file)
                        if voice_data.ndim > 1:
                            voice_data = voice_data[:, 0]  # Convert to mono
                        
                        st.session_state.voice_samples = voice_data
                        st.audio(uploaded_file)
                        st.success(f"Loaded {len(voice_data)/sr:.2f} seconds of audio")
                        
                    except Exception as e:
                        st.error(f"Error reading audio file: {e}")
        
        with col2:
            if len(st.session_state.voice_samples) > 0:
                st.markdown("**Voice Processing Controls**")
                
                # Processing parameters
                voice_gain = st.slider('Voice Gain', 0.1, 3.0, 1.5, 0.1)
                
                # Advanced processing options
                with st.expander("Advanced Processing"):
                    apply_filter = st.checkbox("Apply Low-pass Filter")
                    if apply_filter:
                        cutoff_freq = st.slider('Cutoff Frequency (Hz)', 100, 8000, 3000, 100)
                    
                    apply_compression = st.checkbox("Apply Dynamic Range Compression")
                    if apply_compression:
                        threshold = st.slider('Compression Threshold', 0.1, 1.0, 0.7, 0.1)
                        ratio = st.slider('Compression Ratio', 1.0, 10.0, 4.0, 0.5)
                
                # Process voice
                voice_samples = st.session_state.voice_samples
                processed_voice = voice_samples * voice_gain
                
                # Apply filters if selected
                if apply_filter:
                    from scipy.signal import butter, filtfilt
                    nyquist = sr / 2
                    normalized_cutoff = cutoff_freq / nyquist
                    b, a = butter(4, normalized_cutoff, btype='low')
                    processed_voice = filtfilt(b, a, processed_voice)
                
                # Apply compression if selected
                if apply_compression:
                    # Simple compression algorithm
                    above_threshold = np.abs(processed_voice) > threshold
                    compressed = processed_voice.copy()
                    compressed[above_threshold] = (
                        np.sign(processed_voice[above_threshold]) * 
                        (threshold + (np.abs(processed_voice[above_threshold]) - threshold) / ratio)
                    )
                    processed_voice = compressed
                
                st.session_state.processed_voice = processed_voice
                
                # Play processed audio
                if st.button("🔊 Play Processed Audio"):
                    # Convert to audio bytes for playback
                    buffer = io.BytesIO()
                    sf.write(buffer, processed_voice, sr, format='WAV')
                    audio_bytes = buffer.getvalue()
                    st.audio(audio_bytes, format="audio/wav")
        
        # Voice signal visualization
        if len(st.session_state.voice_samples) > 0:
            st.markdown("**Voice Signal Visualization**")
            
            voice_samples = st.session_state.voice_samples
            processed_voice = st.session_state.processed_voice
            
            # Downsample for visualization if too long
            max_samples = 5000
            if len(voice_samples) > max_samples:
                step = len(voice_samples) // max_samples
                voice_display = voice_samples[::step]
                processed_display = processed_voice[::step]
            else:
                voice_display = voice_samples
                processed_display = processed_voice
            
            # Create comparison plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=voice_display,
                mode='lines',
                name='Original Voice',
                line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                y=processed_display,
                mode='lines',
                name='Processed Voice',
                line=dict(color='red', width=1),
                opacity=0.8
            ))
            fig.update_layout(
                title="Voice Signal Processing Comparison",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Voice signal statistics
            st.markdown("**Voice Signal Statistics**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original RMS", f"{np.sqrt(np.mean(voice_samples**2)):.4f}")
                st.metric("Original Peak", f"{np.max(np.abs(voice_samples)):.4f}")
            
            with col2:
                st.metric("Processed RMS", f"{np.sqrt(np.mean(processed_voice**2)):.4f}")
                st.metric("Processed Peak", f"{np.max(np.abs(processed_voice)):.4f}")
    
    # Sidebar with additional controls
    with st.sidebar:
        st.markdown("### 🛠️ Control Panel")
        
        # Signal history
        if st.session_state.signal_history:
            st.markdown("**Signal History**")
            for i, entry in enumerate(reversed(st.session_state.signal_history[-5:])):
                if st.button(f"{entry['type'].title()} #{len(st.session_state.signal_history)-i}"):
                    st.session_state.current_samples = entry['samples']
                    st.rerun()
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.signal_history = []
            st.session_state.current_samples = np.array([])
            st.session_state.voice_samples = np.array([])
            st.session_state.processed_voice = np.array([])
            st.rerun()
        
        # App info
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This Signal Simulator provides:
        - **Signal Generation**: Create various signal types
        - **Signal Operations**: Apply transformations
        - **Voice Processing**: Record and process audio
        - **Interactive Plots**: Explore your signals
        """)
        
        # Performance metrics
        if len(st.session_state.current_samples) > 0:
            st.markdown("### 📊 Performance")
            st.metric("Samples in Memory", len(st.session_state.current_samples))
            st.metric("Signals Generated", len(st.session_state.signal_history))

if __name__ == "__main__":
    main()
