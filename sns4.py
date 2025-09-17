import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import pandas as pd
import soundfile as sf
import io
import tempfile
import threading
import warnings

warnings.filterwarnings('ignore')

# Try to import audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None

# Try to import sounddevice for playback
try:
    import sounddevice as sd
except ImportError:
    sd = None

# Configure Streamlit page
st.set_page_config(
    page_title="Professional DTS Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state at the very beginning
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'signal_history': [],
        'current_samples': np.array([]),
        'raw_recording': np.array([]),
        'processed_recording': np.array([]),
        'voice_raw': np.array([]),
        'voice_processed': np.array([]),
        'voice_sr': 44100,
        'current_signal_type': 'sine',
        'show_theory': True
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

# Call initialization immediately
initialize_session_state()

# Professional Mature Color Scheme CSS
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for mature color palette */
    :root {
        --primary-dark: #1a365d;
        --primary-medium: #2c5282;
        --primary-light: #3182ce;
        --secondary-dark: #2d3748;
        --secondary-medium: #4a5568;
        --secondary-light: #718096;
        --accent-primary: #2b6cb0;
        --accent-secondary: #3182ce;
        --success-dark: #2f855a;
        --success-medium: #38a169;
        --warning-dark: #d69e2e;
        --warning-medium: #ed8936;
        --error-dark: #c53030;
        --neutral-50: #f7fafc;
        --neutral-100: #edf2f7;
        --neutral-200: #e2e8f0;
        --neutral-700: #4a5568;
        --neutral-800: #2d3748;
        --neutral-900: #1a202c;
    }
    
    /* Override Streamlit default styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--neutral-50);
    }
    
    /* Main headers */
    .main-header {
        font-size: 2.25rem !important;
        font-weight: 700;
        text-align: center;
        color: var(--primary-dark);
        margin-bottom: 2rem;
        letter-spacing: -0.025em;
        text-shadow: none;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem !important;
        font-weight: 600;
        color: var(--secondary-dark);
        margin: 2rem 0 1.5rem 0;
        border-bottom: 2px solid var(--neutral-200);
        padding-bottom: 0.75rem;
        letter-spacing: -0.015em;
    }
    
    /* Professional info boxes */
    .info-box {
        background: linear-gradient(135deg, var(--neutral-50) 0%, #ffffff 100%);
        border: 1px solid var(--neutral-200);
        border-left: 4px solid var(--primary-medium);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .info-box h4 {
        color: var(--primary-dark);
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Warning boxes with mature styling */
    .warning-box {
        background: linear-gradient(135deg, #fef5e7 0%, #fff8f0 100%);
        border: 1px solid #f6e3c5;
        border-left: 4px solid var(--warning-dark);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    .warning-box h3, .warning-box h4 {
        color: var(--warning-dark);
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Success boxes */
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #f7fdf9 100%);
        border: 1px solid #c6f6d5;
        border-left: 4px solid var(--success-dark);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    .success-box h4 {
        color: var(--success-dark);
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Theory cards with professional styling */
    .theory-card {
        background: linear-gradient(135deg, #ffffff 0%, var(--neutral-50) 100%);
        border: 1px solid var(--neutral-200);
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .theory-card h3 {
        color: var(--primary-dark);
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 1px solid var(--neutral-200);
        padding-bottom: 0.5rem;
    }
    
    /* Step indicators with mature design */
    .step-indicator {
        background: linear-gradient(135deg, var(--secondary-dark) 0%, var(--secondary-medium) 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        margin: 1rem 0 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-size: 0.9rem;
        letter-spacing: 0.025em;
    }
    
    /* Override Streamlit button styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-medium) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        letter-spacing: 0.025em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-medium) 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--neutral-100);
        border-right: 1px solid var(--neutral-200);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, var(--neutral-50) 100%);
        border: 1px solid var(--neutral-200);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--neutral-100);
        border: 1px solid var(--neutral-200);
        color: var(--secondary-dark);
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--neutral-100);
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--secondary-medium);
        font-weight: 500;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-medium) 0%, var(--primary-light) 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Professional slider styling */
    .stSlider > div > div > div {
        background-color: var(--primary-medium);
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: var(--neutral-900);
        border: 1px solid var(--neutral-700);
        border-radius: 6px;
    }
    
    /* Override plotly chart backgrounds for consistency */
    .js-plotly-plot {
        border: 1px solid var(--neutral-200);
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

class InteractiveSignalSimulator:
    def __init__(self):
        self.sampling_rate = 44100
        # Professional color palette for plots
        self.colors = {
            'primary': '#2c5282',      # Professional blue
            'secondary': '#2d3748',    # Charcoal gray
            'accent': '#3182ce',       # Medium blue
            'success': '#38a169',      # Forest green
            'warning': '#d69e2e',      # Golden amber
            'error': '#c53030',        # Deep red
            'original': '#2c5282',     # Professional blue
            'processed': '#d69e2e',    # Golden amber
            'voice_original': '#38a169', # Forest green
            'voice_processed': '#c53030' # Deep red
        }

    def get_dts_info(self):
        """Get comprehensive DTS information"""
        return {
            'what_is_dts': """
            **What are Discrete-Time Signals (DTS)?**
            
            Discrete-Time Signals are sequences of numbers that represent information at specific time instances. 
            Unlike continuous signals that exist at every moment, DTS only exist at discrete time points (samples).
            
            **Why are they important?**
            • Digital computers can only process discrete values
            • Enable digital music, photos, videos, and communications  
            • Allow mathematical analysis and precise control
            • Form the basis of all modern digital technology
            
            **Key Characteristics:**
            • **Sampling**: Converting continuous signals to discrete points
            • **Quantization**: Converting continuous amplitudes to discrete levels
            • **Digital Processing**: Mathematical manipulation of discrete samples
            """,
            
            'types_of_dts': """
            **Types of Discrete-Time Signals:**
            
            **1. Periodic Signals** - Repeat their pattern every N samples
            • Example: Pure sine waves, musical notes
            • Formula: x(n) = x(n + N)
            • Used in: Music, communications, signal generators
            
            **2. Aperiodic Signals** - Never repeat their exact pattern  
            • Example: Speech, most natural sounds
            • Formula: x(n) ≠ x(n + N) for any N
            • Used in: Voice recognition, natural sound processing
            
            **3. Deterministic Signals** - Follow a mathematical formula
            • Example: x(n) = sin(2πfn), x(n) = e^(-an)
            • Predictable and can be recreated exactly
            • Used in: Test signals, system analysis
            
            **4. Random Signals** - Cannot be described by exact formulas
            • Example: Noise, stock prices, weather data  
            • Described by statistical properties only
            • Used in: Noise analysis, statistical modeling
            
            **5. Energy Signals** - Have finite total energy
            • Example: Impulse responses, decaying exponentials
            • Total energy = Σ|x(n)|² < ∞
            • Used in: Transient analysis, system responses
            
            **6. Power Signals** - Have finite average power
            • Example: Periodic signals, constant DC signals
            • Average power = finite value over long time
            • Used in: Steady-state analysis, power systems
            """
        }

    def get_operations_info(self):
        """Get detailed information about signal operations"""
        return {
            'amplitude_scale': {
                'title': '📈 Amplitude Scaling',
                'definition': 'Multiplying every sample by a constant factor k',
                'formula': 'y(n) = k × x(n)',
                'effects': {
                    'k > 1': 'Amplification - makes signal louder/stronger',
                    'k = 1': 'No change - identity operation', 
                    '0 < k < 1': 'Attenuation - makes signal quieter/weaker',
                    'k = 0': 'Silence - signal becomes zero',
                    'k < 0': 'Inversion - flips signal upside down'
                },
                'applications': [
                    'Volume control in audio systems',
                    'Brightness adjustment in image processing',
                    'Automatic gain control in communications',
                    'Audio mixing and mastering',
                    'Sensor calibration and measurement'
                ],
                'example': 'When adjusting volume from 50% to 100%, the audio signal is scaled by k=2.0'
            },
            
            'amplitude_shift': {
                'title': '⬆️ Amplitude Shift (DC Offset)',
                'definition': 'Adding a constant value c to every sample',
                'formula': 'y(n) = x(n) + c',
                'effects': {
                    'c > 0': 'Positive offset - shifts signal upward',
                    'c = 0': 'No change - identity operation',
                    'c < 0': 'Negative offset - shifts signal downward'
                },
                'applications': [
                    'Brightness adjustment in digital imaging',
                    'Sensor bias removal in measurements', 
                    'DC restoration in video processing',
                    'Baseline correction in scientific data',
                    'Battery voltage compensation circuits'
                ],
                'example': 'Image brightness adjustment adds the same value to every pixel, lifting dark regions'
            },
            
            'time_scale': {
                'title': '⏩ Time Scaling', 
                'definition': 'Compressing or expanding the signal along time axis',
                'formula': 'y(n) = x(αn) where α is scaling factor',
                'effects': {
                    'α > 1': 'Compression - faster playback, higher pitch',
                    'α = 1': 'No change - normal speed',
                    '0 < α < 1': 'Expansion - slower playback, lower pitch'
                },
                'applications': [
                    'Video speed control and time-lapse photography',
                    'Audio pitch shifting for music production',
                    'Radar pulse compression in defense systems', 
                    'Heart rate analysis in medical diagnostics',
                    'Speech rate modification for accessibility'
                ],
                'example': '2x video playback compresses time by α=2, creating fast-forward effect with pitch change'
            },
            
            'time_shift': {
                'title': '⏰ Time Shift (Delay)',
                'definition': 'Moving the signal forward or backward in time',
                'formula': 'y(n) = x(n - k) where k is shift amount',
                'effects': {
                    'k > 0': 'Delay - signal appears later in time',
                    'k = 0': 'No change - no temporal shift', 
                    'k < 0': 'Advance - signal appears earlier in time'
                },
                'applications': [
                    'Echo and reverb effects in audio production',
                    'Signal synchronization in communication systems',
                    'Distance measurement in radar and GPS',
                    'Delay compensation in live audio systems',
                    'Multi-sensor data alignment in research'
                ],
                'example': 'Audio echo: original voice at t=0, echo at t=0.5s (22,050 sample delay at 44.1kHz)'
            }
        }

    def create_demo_signals(self, operation_type):
        """Create demonstration signals for operations"""
        n = np.arange(100)
        base_signal = np.sin(2 * np.pi * n / 20)  # Base sine wave
        
        demos = []
        
        if operation_type == 'amplitude_scale':
            scales = [0.5, 1.0, 2.0, -1.0]
            labels = ['Half (0.5×)', 'Original (1.0×)', 'Double (2.0×)', 'Inverted (-1.0×)']
            for scale, label in zip(scales, labels):
                demos.append((base_signal * scale, label))
                
        elif operation_type == 'amplitude_shift':
            shifts = [-1.0, 0.0, 1.0, 2.0]
            labels = ['Shifted Down (-1)', 'Original (0)', 'Shifted Up (+1)', 'Large Offset (+2)']
            for shift, label in zip(shifts, labels):
                demos.append((base_signal + shift, label))
                
        elif operation_type == 'time_scale':
            scales = [0.5, 1.0, 2.0]
            labels = ['Expanded (0.5×)', 'Original (1.0×)', 'Compressed (2.0×)']
            for scale, label in zip(scales, labels):
                new_n = n * scale
                scaled_signal = np.interp(n, new_n, base_signal, left=0, right=0)
                demos.append((scaled_signal, label))
                
        elif operation_type == 'time_shift':
            shifts = [-10, 0, 10, 20]
            labels = ['Advanced (-10)', 'Original (0)', 'Delayed (+10)', 'Large Delay (+20)']
            for shift, label in zip(shifts, labels):
                shifted_signal = np.roll(base_signal, shift)
                demos.append((shifted_signal, label))
        
        return demos

    def plot_demo(self, demos):
        """Plot operation demonstrations with professional styling"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[demo[1] for demo in demos[:4]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Professional color palette
        colors = [self.colors['primary'], self.colors['success'], self.colors['warning'], self.colors['error']]
        
        for i, (signal, label) in enumerate(demos[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    y=signal,
                    mode='lines',
                    name=label,
                    line=dict(color=colors[i], width=2.5),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=500,
            title_text="Interactive Operation Demonstration",
            template="simple_white",
            font=dict(family="Inter, sans-serif", size=12, color=self.colors['secondary']),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig

    def generate_signal(self, signal_type, params):
        """Generate different types of signals"""
        try:
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
                    white = np.random.randn(length)
                    f = np.fft.fftfreq(length)
                    f = np.abs(f)
                    f[0] = 1e-6
                    pink_filter = 1 / np.sqrt(f)
                    pink_filter[0] = 0
                    samples = amp * np.fft.ifft(np.fft.fft(white) * pink_filter).real
            else:
                samples = np.zeros(length)

            return samples.astype(np.float64)
            
        except Exception as e:
            st.error(f"Error generating signal: {str(e)}")
            return np.zeros(1000)

    def apply_operations(self, samples, operations):
        """Apply signal operations with enhanced processing"""
        if len(samples) == 0:
            return samples
            
        try:
            processed = samples.copy().astype(np.float64)
            
            # Apply operations in sequence
            processed = processed * operations.get('amplitude_scale', 1.0)
            processed = processed + operations.get('amplitude_shift', 0.0)
            
            time_scale_val = operations.get('time_scale', 1.0)
            if time_scale_val != 1.0:
                length = len(processed)
                x = np.arange(length)
                new_x = x * time_scale_val
                processed = np.interp(x, new_x, processed, left=0, right=0)
            
            time_shift = int(operations.get('time_shift', 0))
            if time_shift != 0:
                processed = np.roll(processed, time_shift)
                
            return processed
            
        except Exception as e:
            st.error(f"Error applying operations: {str(e)}")
            return samples

    def periodicity_check(self, x):
        """Enhanced periodicity analysis for LTI systems"""
        if x is None or len(x) < 50:
            return 0.0, "Signal too short", "Need at least 50 samples"
        
        try:
            # Remove DC component
            x_centered = x - np.mean(x)
            
            # Autocorrelation analysis
            f = np.fft.fft(x_centered, n=2*len(x_centered))
            ac = np.fft.ifft(f * np.conjugate(f)).real
            ac = ac[:len(x_centered)]
            ac /= (ac[0] + 1e-12)
            
            # Find peaks in autocorrelation
            score = np.max(np.abs(ac[1:len(ac)//2]))
            
            # Enhanced classification
            if score > 0.8:
                verdict = "Highly Periodic"
                explanation = "Strong repetitive pattern detected"
            elif score > 0.5:
                verdict = "Moderately Periodic"
                explanation = "Some repetitive structure present"
            elif score > 0.3:
                verdict = "Weakly Periodic"
                explanation = "Minor periodic components detected"
            else:
                verdict = "Non-Periodic"
                explanation = "No significant repetitive pattern"
            
            return float(score), verdict, explanation
        
        except Exception as e:
            return 0.0, "Analysis Failed", f"Error: {str(e)}"

    def plot_signal_plotly(self, samples, title="Signal", color_key="primary"):
        """Enhanced plotting with professional styling"""
        fig = go.Figure()
        
        if len(samples) > 0:
            time_axis = np.arange(len(samples)) / self.sampling_rate
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=samples,
                mode='lines',
                name=title,
                line=dict(color=self.colors.get(color_key, self.colors['primary']), width=2.5),
                hovertemplate='<b>Time</b>: %{x:.4f}s<br><b>Amplitude</b>: %{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': self.colors['secondary'], 'family': 'Inter, sans-serif'}
            },
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="simple_white",
            height=400,
            showlegend=False,
            hovermode='x unified',
            font=dict(family="Inter, sans-serif", size=12, color=self.colors['secondary']),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig

    def plot_comparison_plotly(self, original, processed):
        """Enhanced comparison plot with professional styling"""
        fig = go.Figure()
        
        time_orig = np.arange(len(original)) / self.sampling_rate
        time_proc = np.arange(len(processed)) / self.sampling_rate
        
        fig.add_trace(go.Scatter(
            x=time_orig,
            y=original,
            mode='lines',
            name='Original Signal',
            line=dict(color=self.colors['original'], width=2.5),
            hovertemplate='<b>Original</b><br>Time: %{x:.4f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=time_proc,
            y=processed,
            mode='lines',
            name='Processed Signal',
            line=dict(color=self.colors['processed'], width=2.5),
            opacity=0.85,
            hovertemplate='<b>Processed</b><br>Time: %{x:.4f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Signal Comparison: Original vs Processed",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': self.colors['secondary'], 'family': 'Inter, sans-serif'}
            },
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="simple_white",
            height=450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=self.colors['secondary'],
                borderwidth=1
            ),
            hovermode='x unified',
            font=dict(family="Inter, sans-serif", size=12, color=self.colors['secondary']),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig

def signal_generator_page():
    """Enhanced Signal Generator with professional design"""
    simulator = InteractiveSignalSimulator()
    
    st.markdown('<p class="main-header">Professional DTS Analysis Platform</p>', unsafe_allow_html=True)
    
    # DTS Theory Section
    if st.session_state.show_theory:
        st.markdown('<p class="section-header">Discrete-Time Signal Theory</p>', unsafe_allow_html=True)
        
        dts_info = simulator.get_dts_info()
        
        # What is DTS
        with st.expander("🔬 Fundamentals of Discrete-Time Signals", expanded=True):
            st.markdown(dts_info['what_is_dts'])
        
        # Types of DTS
        with st.expander("📊 Classification of DTS Types", expanded=False):
            st.markdown(dts_info['types_of_dts'])
        
        # Real-world examples
        st.markdown("**Industry Applications of DTS:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Audio Engineering**\nDigital audio workstations, streaming platforms, and professional recording systems")
        with col2:
            st.info("**Telecommunications**\nMobile networks, satellite communications, and data transmission protocols")
        with col3:
            st.info("**Medical Technology**\nDiagnostic imaging, patient monitoring systems, and biomedical signal analysis")

    # Signal Operations Theory
    st.markdown('<p class="section-header">Signal Processing Operations</p>', unsafe_allow_html=True)
    
    operations_info = simulator.get_operations_info()
    
    # Create tabs for each operation with professional styling
    tab1, tab2, tab3, tab4 = st.tabs(["Amplitude Scaling", "Amplitude Shift", "Time Scaling", "Time Shift"])
    
    for tab, op_key in zip([tab1, tab2, tab3, tab4], ['amplitude_scale', 'amplitude_shift', 'time_scale', 'time_shift']):
        with tab:
            op_info = operations_info[op_key]
            
            # Operation explanation with professional layout
            col_theory, col_demo = st.columns([1, 1])
            
            with col_theory:
                st.markdown(f"### {op_info['title']}")
                st.markdown(f"**Definition:** {op_info['definition']}")
                
                # Mathematical formula
                st.code(op_info['formula'], language='python')
                
                # Parameter effects
                st.markdown("**Parameter Effects:**")
                for param, effect in op_info['effects'].items():
                    st.markdown(f"• **{param}**: {effect}")
                
                # Applications
                st.markdown("**Professional Applications:**")
                for app in op_info['applications']:
                    st.markdown(f"• {app}")
                
                # Example
                st.markdown("**Practical Example:**")
                st.markdown(f'<div class="success-box">{op_info["example"]}</div>', unsafe_allow_html=True)
            
            with col_demo:
                # Interactive demo
                if st.button(f"View Interactive Demo", key=f"demo_{op_key}"):
                    demos = simulator.create_demo_signals(op_key)
                    fig = simulator.plot_demo(demos)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('<div class="info-box">The visualization demonstrates how different parameter values affect the same base signal, illustrating the mathematical relationship between input and output.</div>', unsafe_allow_html=True)

    # Practical Signal Generation
    st.markdown('<p class="section-header">Signal Generation Workshop</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown('<div class="step-indicator">Step 1: Signal Configuration</div>', unsafe_allow_html=True)
        
        signal_type = st.selectbox(
            "Signal Type",
            ['sine', 'impulse', 'step', 'exponential', 'chirp', 'noise'],
            index=0,
            help="Select the mathematical signal type to generate"
        )
        
        # Professional signal information
        signal_descriptions = {
            'sine': "**Sinusoidal Wave:** Fundamental harmonic oscillation used in frequency domain analysis",
            'impulse': "**Impulse Function:** Dirac delta approximation for system response characterization",
            'step': "**Unit Step Function:** Heaviside function modeling sudden system changes",
            'exponential': "**Exponential Decay:** Mathematical model for natural decay processes",
            'chirp': "**Frequency Sweep:** Linear frequency modulation for system identification",
            'noise': "**Stochastic Signal:** Random process for statistical analysis and testing"
        }
        
        if signal_type in signal_descriptions:
            st.markdown(f'<div class="info-box">{signal_descriptions[signal_type]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-indicator">Step 2: Parameter Configuration</div>', unsafe_allow_html=True)
        
        # Professional parameter layout
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            amplitude = st.slider('Signal Amplitude', 0.1, 10.0, 1.0, 0.1, help="Peak amplitude value")
            length = st.slider('Sample Length', 100, 10000, 1000, 100, help="Number of discrete samples")
        
        with param_col2:
            params = {'amplitude': amplitude, 'length': length}
            
            if signal_type in ['impulse', 'step']:
                params['position'] = st.slider('Event Position', 0, length-1, length//4, help="Sample index of event occurrence")
            elif signal_type == 'sine':
                params['frequency'] = st.slider('Frequency (Hz)', 0.1, 100.0, 5.0, 0.1, help="Oscillation frequency")
                params['phase'] = st.slider('Phase (radians)', -np.pi, np.pi, 0.0, 0.1, help="Initial phase offset")
            elif signal_type == 'exponential':
                params['decay'] = st.slider('Decay Constant', 0.001, 1.0, 0.01, 0.001, help="Rate of exponential decay")
            elif signal_type == 'chirp':
                params['f0'] = st.slider('Start Frequency (Hz)', 0.1, 50.0, 1.0, 0.1)
                params['f1'] = st.slider('End Frequency (Hz)', 0.1, 50.0, 20.0, 0.1)
                params['t1'] = st.slider('Sweep Duration (s)', 0.1, 5.0, 1.0, 0.1)
            elif signal_type == 'noise':
                params['noise_type'] = st.selectbox('Noise Spectrum', ['white', 'pink'], help="Noise frequency characteristics")
        
        st.markdown('<div class="step-indicator">Step 3: Signal Generation</div>', unsafe_allow_html=True)
        
        if st.button("Generate Professional Signal", type="primary", use_container_width=True):
            with st.spinner('Processing signal generation...'):
                samples = simulator.generate_signal(signal_type, params)
                st.session_state.current_samples = samples
                st.session_state.current_signal_type = signal_type
                st.session_state.signal_history.append({
                    'type': signal_type,
                    'params': params.copy(),
                    'samples': samples.copy()
                })
            
            st.success(f"✓ Generated {signal_type} signal with {len(samples):,} samples")
    
    with col2:
        if len(st.session_state.current_samples) > 0:
            samples = st.session_state.current_samples
            
            fig = simulator.plot_signal_plotly(samples, f"{signal_type.title()} Signal", "primary")
            st.plotly_chart(fig, use_container_width=True)
            
            # Professional signal analysis
            st.markdown("**Signal Analysis**")
            
            analysis_col1, analysis_col2 = st.columns(2)
            with analysis_col1:
                st.metric("Sample Count", f"{len(samples):,}")
                st.metric("RMS Value", f"{np.sqrt(np.mean(samples**2)):.4f}")
                st.metric("Peak Amplitude", f"{np.max(np.abs(samples)):.4f}")
            
            with analysis_col2:
                st.metric("Time Duration", f"{len(samples)/44100:.3f}s")
                st.metric("Mean Value", f"{np.mean(samples):.4f}")
                st.metric("Standard Deviation", f"{np.std(samples):.4f}")
        else:
            st.markdown('<div class="warning-box"><h4>Signal Generation Required</h4><p>Configure and generate a signal to view analysis and processing options.</p></div>', unsafe_allow_html=True)

    # Professional Signal Operations Application
    if len(st.session_state.current_samples) > 0:
        st.markdown('<p class="section-header">Signal Processing Operations</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-indicator">Step 4: Apply Signal Transformations</div>', unsafe_allow_html=True)
        
        # Professional operations interface
        op_col1, op_col2 = st.columns(2)
        
        with op_col1:
            st.markdown("**Amplitude Domain Operations**")
            amplitude_scale = st.slider('Amplitude Scaling Factor', 0.1, 5.0, 1.0, 0.1, key="main_amp_scale")
            amplitude_shift = st.slider('DC Offset Value', -3.0, 3.0, 0.0, 0.1, key="main_amp_shift")
        
        with op_col2:
            st.markdown("**Time Domain Operations**")
            time_scale = st.slider('Time Scaling Factor', 0.5, 2.0, 1.0, 0.1, key="main_time_scale")
            time_shift = st.slider('Temporal Shift (samples)', -200, 200, 0, 1, key="main_time_shift")
        
        # Apply operations with professional feedback
        operations = {
            'amplitude_scale': amplitude_scale,
            'amplitude_shift': amplitude_shift,
            'time_scale': time_scale,
            'time_shift': time_shift
        }
        
        processed_samples = simulator.apply_operations(st.session_state.current_samples, operations)
        
        # Professional operation summary
        transformations = []
        if amplitude_scale != 1.0:
            transformations.append(f"Amplitude scaling: {amplitude_scale:.1f}×")
        if amplitude_shift != 0.0:
            transformations.append(f"DC offset: {amplitude_shift:+.2f}")
        if time_scale != 1.0:
            transformations.append(f"Time scaling: {time_scale:.1f}×")
        if time_shift != 0:
            transformations.append(f"Temporal shift: {time_shift} samples")
        
        if transformations:
            st.info("**Applied Transformations:** " + " | ".join(transformations))
        else:
            st.info("**Status:** No transformations applied - all parameters at identity values")
        
        # Professional comparison visualization
        fig_comp = simulator.plot_comparison_plotly(st.session_state.current_samples, processed_samples)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Professional export options
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            df_orig = pd.DataFrame({
                'Sample_Index': range(len(st.session_state.current_samples)),
                'Time_s': np.arange(len(st.session_state.current_samples)) / 44100,
                'Amplitude': st.session_state.current_samples
            })
            csv_orig = df_orig.to_csv(index=False)
            st.download_button(
                "Export Original Signal Data",
                data=csv_orig,
                file_name=f"original_{signal_type}_signal.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # Ensure data consistency for export
            orig_len = len(st.session_state.current_samples)
            proc_len = len(processed_samples)
            
            if proc_len <= orig_len:
                original_padded = st.session_state.current_samples[:proc_len]
            else:
                original_padded = np.pad(st.session_state.current_samples, (0, proc_len - orig_len))
            
            df_proc = pd.DataFrame({
                'Sample_Index': range(len(processed_samples)),
                'Time_s': np.arange(len(processed_samples)) / 44100,
                'Original': original_padded,
                'Processed': processed_samples
            })
            csv_proc = df_proc.to_csv(index=False)
            st.download_button(
                "Export Processed Signal Data",
                data=csv_proc,
                file_name=f"processed_{signal_type}_signal.csv",
                mime="text/csv",
                use_container_width=True
            )

def voice_dts_processor_page():
    """Professional Voice DTS Processor"""
    simulator = InteractiveSignalSimulator()
    
    st.markdown('<p class="main-header">Voice Signal Analysis Laboratory</p>', unsafe_allow_html=True)
    
    # Professional educational introduction
    if st.session_state.show_theory:
        with st.expander("🔬 Voice Signal Processing Fundamentals", expanded=True):
            st.markdown("""
            **Analog-to-Digital Conversion Process:**
            
            1. **Acoustic Capture**: Microphone converts sound pressure waves to electrical signals
            2. **Anti-Aliasing Filtering**: Removes frequencies above Nyquist limit (fs/2)
            3. **Sampling**: Discrete-time sampling at regular intervals (typically 44.1kHz)
            4. **Quantization**: Amplitude discretization to finite precision levels
            5. **Digital Processing**: Mathematical manipulation of discrete signal samples
            
            **Professional Applications:**
            - Speech recognition and natural language processing
            - Audio compression algorithms (MP3, AAC, Opus)
            - Real-time communication systems
            - Acoustic signal enhancement and noise reduction
            """)
        
        with st.expander("⚙️ Linear Time-Invariant System Analysis", expanded=False):
            st.markdown("""
            **LTI System Properties:**
            
            - **Linearity**: Superposition principle applies - system response scales proportionally
            - **Time-Invariance**: System characteristics remain constant over time
            - **Stability**: Bounded input produces bounded output (BIBO stability)
            - **Causality**: Output depends only on present and past inputs
            
            **Periodicity Analysis Metrics:**
            - **Score > 0.8**: Highly periodic signals (sustained vowels, musical tones)
            - **Score 0.5-0.8**: Moderately periodic (speech with prosodic patterns)
            - **Score < 0.5**: Aperiodic signals (fricatives, noise, random speech)
            """)

    # Professional main interface
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown('<div class="step-indicator">Step 1: Audio Acquisition</div>', unsafe_allow_html=True)
        
        # Professional recording interface
        acquisition_method = st.radio(
            "Select acquisition method:",
            ["🎙️ Real-time Recording", "📂 File Import"],
            horizontal=True
        )
        
        if acquisition_method == "🎙️ Real-time Recording":
            st.markdown("**Professional Audio Recording:**")
            
            if audio_recorder is None:
                st.error("⚠️ Audio recording component unavailable. Install: `pip install audio-recorder-streamlit`")
                st.info("💡 **Alternative:** Use file import option below")
            else:
                try:
                    audio_bytes = audio_recorder(
                        text="Start Recording",
                        recording_color="#c53030",
                        neutral_color="#2c5282",
                        icon_name="microphone",
                        icon_size="1.5x",
                        pause_threshold=1.5,
                        sample_rate=44100
                    )
                    
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                        
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(audio_bytes)
                                tmp_file.flush()
                                
                                voice_data, sr = sf.read(tmp_file.name)
                                if voice_data.ndim > 1:
                                    voice_data = voice_data[:, 0]  # Convert to mono
                                
                                st.session_state.voice_raw = voice_data
                                st.session_state.voice_sr = sr
                                
                                # Professional analysis
                                score, verdict, explanation = simulator.periodicity_check(voice_data)
                                
                                st.success(f"✓ Audio acquisition completed successfully")
                                st.info(f"**Signal Properties:** {len(voice_data)/sr:.2f}s duration, {sr}Hz sample rate")
                                
                                # Professional LTI analysis display
                                st.markdown(f"""
                                <div class='info-box'>
                                    <h4>LTI System Analysis Results</h4>
                                    <p><strong>Classification:</strong> {verdict}</p>
                                    <p><strong>Periodicity Score:</strong> {score:.3f}</p>
                                    <p><strong>Analysis:</strong> {explanation}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"⚠️ Audio processing error: {e}")
                            
                except Exception as e:
                    st.error(f"⚠️ Recording system error: {e}")
        
        else:  # File import
            st.markdown("**Professional Audio Import:**")
            uploaded_file = st.file_uploader(
                "Select audio file for analysis",
                type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
                help="Supported formats: WAV, MP3, OGG, FLAC, M4A (professional codecs)"
            )
            
            if uploaded_file:
                try:
                    voice_data, sr = sf.read(uploaded_file)
                    if voice_data.ndim > 1:
                        voice_data = voice_data[:, 0]
                    
                    st.session_state.voice_raw = voice_data
                    st.session_state.voice_sr = sr
                    
                    st.audio(uploaded_file)
                    
                    # Professional analysis
                    score, verdict, explanation = simulator.periodicity_check(voice_data)
                    
                    st.success(f"✓ Audio file imported successfully")
                    st.info(f"**Signal Properties:** {len(voice_data)/sr:.2f}s duration, {sr}Hz sample rate")
                    
                    # Professional LTI analysis
                    st.markdown(f"""
                    <div class='info-box'>
                        <h4>LTI System Analysis Results</h4>
                        <p><strong>Classification:</strong> {verdict}</p>
                        <p><strong>Periodicity Score:</strong> {score:.3f}</p>
                        <p><strong>Analysis:</strong> {explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"⚠️ File processing error: {e}")
    
    with col2:
        st.markdown("**Professional Guidelines**")
        
        st.markdown("""
        <div class='info-box'>
            <h4>Recording Best Practices:</h4>
            <ul>
                <li>Maintain consistent 6-inch microphone distance</li>
                <li>Ensure ambient noise below -40dB</li>
                <li>Use controlled acoustic environment</li>
                <li>Speak at consistent volume levels</li>
                <li>Avoid clipping (peak levels > -6dBFS)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4>Analysis Test Signals:</h4>
            <ul>
                <li><strong>Sustained "Ahhh"</strong> - High periodicity (0.8+)</li>
                <li><strong>Musical whistle</strong> - Very high periodicity (0.9+)</li>
                <li><strong>Counting sequence</strong> - Medium periodicity (0.3-0.6)</li>
                <li><strong>White noise</strong> - Low periodicity (<0.2)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Professional signal visualization and processing
    if len(st.session_state.voice_raw) > 0:
        st.markdown('<div class="step-indicator">Step 2: Signal Analysis</div>', unsafe_allow_html=True)
        
        # Professional original signal visualization
        fig_orig = simulator.plot_signal_plotly(
            st.session_state.voice_raw,
            "Original Voice Signal (Discrete-Time Domain)",
            "voice_original"
        )
        st.plotly_chart(fig_orig, use_container_width=True)
        
        # Professional signal processing section
        st.markdown('<div class="step-indicator">Step 3: Digital Signal Processing</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <strong>Real-time Processing Controls:</strong> Adjust parameters to observe real-time effects on signal characteristics and spectral content.
        </div>
        """, unsafe_allow_html=True)
        
        # Professional processing controls
        proc_col1, proc_col2 = st.columns(2)
        
        with proc_col1:
            st.markdown("**Amplitude Domain Processing**")
            
            amp_scale = st.slider(
                'Amplitude Scaling Factor',
                0.1, 4.0, 1.0, 0.1,
                key="voice_amp_scale",
                help="Linear amplitude multiplication (gain control)"
            )
            
            amp_shift = st.slider(
                'DC Offset Adjustment',
                -1.0, 1.0, 0.0, 0.1,
                key="voice_amp_shift",
                help="Constant amplitude offset (baseline correction)"
            )
        
        with proc_col2:
            st.markdown("**Time Domain Processing**")
            
            time_scale = st.slider(
                'Temporal Scaling Factor',
                0.5, 2.0, 1.0, 0.1,
                key="voice_time_scale",
                help="Time-axis compression/expansion (playback speed)"
            )
            
            time_shift = st.slider(
                'Temporal Offset (samples)',
                -500, 500, 0, 10,
                key="voice_time_shift",
                help="Time-domain delay/advance (synchronization)"
            )
        
        # Professional real-time processing
        voice_operations = {
            'amplitude_scale': amp_scale,
            'amplitude_shift': amp_shift,
            'time_scale': time_scale,
            'time_shift': time_shift
        }
        
        processed_voice = simulator.apply_operations(st.session_state.voice_raw, voice_operations)
        st.session_state.voice_processed = processed_voice
        
        # Professional operation summary
        processing_operations = []
        if amp_scale != 1.0:
            processing_operations.append(f"Amplitude scaling: {amp_scale:.2f}× ({20*np.log10(amp_scale):+.1f}dB)")
        if amp_shift != 0.0:
            processing_operations.append(f"DC offset: {amp_shift:+.3f}")
        if time_scale != 1.0:
            processing_operations.append(f"Time scaling: {time_scale:.2f}× ({1/time_scale:.2f}× duration)")
        if time_shift != 0:
            processing_operations.append(f"Temporal shift: {time_shift} samples ({time_shift/44100*1000:.1f}ms)")
        
        if processing_operations:
            st.info("**Active Processing Operations:** " + " | ".join(processing_operations))
        else:
            st.info("**Processing Status:** Identity transform - no modifications applied")
        
        # Professional processed signal visualization
        st.markdown('<div class="step-indicator">Step 4: Processed Signal Analysis</div>', unsafe_allow_html=True)
        
        fig_proc = simulator.plot_signal_plotly(
            processed_voice,
            "Processed Voice Signal (Post-Processing)",
            "voice_processed"
        )
        st.plotly_chart(fig_proc, use_container_width=True)
        
        # Professional signal comparison
        st.markdown('<div class="step-indicator">Step 5: Comparative Analysis</div>', unsafe_allow_html=True)
        
        # Use professional colors for comparison
        fig_comp = go.Figure()
        
        time_orig = np.arange(len(st.session_state.voice_raw)) / st.session_state.voice_sr
        time_proc = np.arange(len(processed_voice)) / st.session_state.voice_sr
        
        fig_comp.add_trace(go.Scatter(
            x=time_orig,
            y=st.session_state.voice_raw,
            mode='lines',
            name='Original Signal',
            line=dict(color=simulator.colors['voice_original'], width=2.5),
        ))
        
        fig_comp.add_trace(go.Scatter(
            x=time_proc,
            y=processed_voice,
            mode='lines',
            name='Processed Signal',
            line=dict(color=simulator.colors['voice_processed'], width=2.5),
            opacity=0.85,
        ))
        
        fig_comp.update_layout(
            title={
                'text': "Comparative Signal Analysis: Original vs Processed",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': simulator.colors['secondary'], 'family': 'Inter, sans-serif'}
            },
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="simple_white",
            height=450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=simulator.colors['secondary'],
                borderwidth=1
            ),
            font=dict(family="Inter, sans-serif", size=12, color=simulator.colors['secondary']),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Professional analysis and playback section
        st.markdown('<div class="step-indicator">Step 6: Performance Analysis & Export</div>', unsafe_allow_html=True)
        
        # Professional statistical analysis
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.markdown("**Original Signal Metrics**")
            orig_rms = np.sqrt(np.mean(st.session_state.voice_raw**2))
            orig_peak = np.max(np.abs(st.session_state.voice_raw))
            orig_duration = len(st.session_state.voice_raw) / st.session_state.voice_sr
            orig_crest = orig_peak / orig_rms if orig_rms > 0 else 0
            
            st.metric("RMS Level", f"{orig_rms:.4f}")
            st.metric("Peak Level", f"{orig_peak:.4f}")
            st.metric("Duration", f"{orig_duration:.3f}s")
            st.metric("Crest Factor", f"{orig_crest:.2f}")
        
        with analysis_col2:
            st.markdown("**Processed Signal Metrics**")
            proc_rms = np.sqrt(np.mean(processed_voice**2))
            proc_peak = np.max(np.abs(processed_voice))
            proc_duration = len(processed_voice) / st.session_state.voice_sr
            proc_crest = proc_peak / proc_rms if proc_rms > 0 else 0
            
            st.metric("RMS Level", f"{proc_rms:.4f}")
            st.metric("Peak Level", f"{proc_peak:.4f}")
            st.metric("Duration", f"{proc_duration:.3f}s")
            st.metric("Crest Factor", f"{proc_crest:.2f}")
        
        with analysis_col3:
            st.markdown("**Processing Analysis**")
            rms_ratio = (proc_rms / orig_rms) if orig_rms > 0 else 0
            peak_ratio = (proc_peak / orig_peak) if orig_peak > 0 else 0
            duration_ratio = (proc_duration / orig_duration) if orig_duration > 0 else 0
            
            st.metric("RMS Change", f"{rms_ratio:.3f}× ({20*np.log10(rms_ratio) if rms_ratio > 0 else -np.inf:.1f}dB)")
            st.metric("Peak Change", f"{peak_ratio:.3f}×")
            st.metric("Duration Change", f"{duration_ratio:.3f}×")
        
        # Professional playback and export controls
        st.markdown("**Audio Playback & Export**")
        
        playback_col1, playback_col2, playback_col3 = st.columns(3)
        
        with playback_col1:
            if st.button("🔊 Original Signal Playback", use_container_width=True):
                try:
                    buffer = io.BytesIO()
                    if np.max(np.abs(st.session_state.voice_raw)) > 0:
                        normalized_orig = st.session_state.voice_raw / np.max(np.abs(st.session_state.voice_raw)) * 0.9
                    else:
                        normalized_orig = st.session_state.voice_raw
                    sf.write(buffer, normalized_orig, st.session_state.voice_sr, format='WAV')
                    st.audio(buffer.getvalue(), format="audio/wav")
                except Exception as e:
                    st.error(f"Playback error: {e}")
        
        with playback_col2:
            if st.button("🎵 Processed Signal Playback", use_container_width=True):
                try:
                    buffer = io.BytesIO()
                    if np.max(np.abs(processed_voice)) > 0:
                        normalized_proc = processed_voice / np.max(np.abs(processed_voice)) * 0.9
                    else:
                        normalized_proc = processed_voice
                    sf.write(buffer, normalized_proc, st.session_state.voice_sr, format='WAV')
                    st.audio(buffer.getvalue(), format="audio/wav")
                except Exception as e:
                    st.error(f"Playback error: {e}")
        
        with playback_col3:
            if st.button("💾 Professional Export", use_container_width=True):
                try:
                    buffer = io.BytesIO()
                    if np.max(np.abs(processed_voice)) > 0:
                        normalized_proc = processed_voice / np.max(np.abs(processed_voice)) * 0.9
                    else:
                        normalized_proc = processed_voice
                    sf.write(buffer, normalized_proc, st.session_state.voice_sr, format='WAV', subtype='PCM_24')
                    
                    st.download_button(
                        label="Download Professional WAV (24-bit)",
                        data=buffer.getvalue(),
                        file_name="processed_voice_professional.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        # Professional advanced analysis
        with st.expander("🔬 Advanced Signal Analysis", expanded=False):
            st.markdown("**Detailed Signal Characteristics:**")
            
            advanced_col1, advanced_col2 = st.columns(2)
            
            with advanced_col1:
                st.markdown("**Original Signal Properties:**")
                orig_mean = np.mean(st.session_state.voice_raw)
                orig_std = np.std(st.session_state.voice_raw)
                orig_energy = np.sum(st.session_state.voice_raw**2)
                orig_power = orig_energy / len(st.session_state.voice_raw)
                
                st.text(f"DC Component: {orig_mean:.6f}")
                st.text(f"Standard Deviation: {orig_std:.4f}")
                st.text(f"Signal Energy: {orig_energy:.2f}")
                st.text(f"Average Power: {orig_power:.6f}")
                st.text(f"Dynamic Range: {20*np.log10(orig_peak/orig_rms) if orig_rms > 0 else 0:.1f} dB")
            
            with advanced_col2:
                st.markdown("**Processed Signal Properties:**")
                proc_mean = np.mean(processed_voice)
                proc_std = np.std(processed_voice)
                proc_energy = np.sum(processed_voice**2)
                proc_power = proc_energy / len(processed_voice)
                
                st.text(f"DC Component: {proc_mean:.6f}")
                st.text(f"Standard Deviation: {proc_std:.4f}")
                st.text(f"Signal Energy: {proc_energy:.2f}")
                st.text(f"Average Power: {proc_power:.6f}")
                st.text(f"Dynamic Range: {20*np.log10(proc_peak/proc_rms) if proc_rms > 0 else 0:.1f} dB")
    
    else:
        # Professional standby interface
        st.markdown("""
        <div class='warning-box'>
            <h3>Professional Voice Analysis System</h3>
            <p>Configure audio acquisition above to begin professional voice signal analysis and discrete-time processing.</p>
            <p><strong>System Capabilities:</strong></p>
            <ul>
                <li>High-fidelity voice signal acquisition and digitization</li>
                <li>Real-time discrete-time signal processing operations</li>
                <li>Advanced LTI system analysis with periodicity measurement</li>
                <li>Professional-grade signal visualization and analysis</li>
                <li>Comparative analysis with statistical metrics</li>
                <li>High-quality audio export with professional codecs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Professional main application interface"""
    
    with st.sidebar:
        st.markdown("### 🎛️ Professional Control Center")
        
        page = st.radio(
            "Select Analysis Module:",
            ["📊 Signal Generation Laboratory", "🎤 Voice Analysis Laboratory"],
            index=0,
            help="Choose your professional analysis environment"
        )
        
        st.markdown("---")
        
        # Professional controls
        st.session_state.show_theory = st.checkbox(
            "📚 Display Theoretical Framework",
            value=st.session_state.show_theory,
            help="Show/hide educational and theoretical content"
        )
        
        # Professional module-specific controls
        if page == "📊 Signal Generation Laboratory":
            st.markdown("**Signal Generation Controls**")
            
            if st.session_state.signal_history:
                st.markdown("**Recent Signal Library**")
                for i, entry in enumerate(reversed(st.session_state.signal_history[-3:])):
                    idx = len(st.session_state.signal_history) - i
                    if st.button(f"Load {entry['type'].title()} #{idx}", key=f"load_{idx}"):
                        st.session_state.current_samples = entry['samples']
                        st.session_state.current_signal_type = entry['type']
                        st.rerun()
            
            if st.button("Generate Random Test Signal"):
                import random
                signal_types = ['sine', 'impulse', 'step', 'exponential', 'chirp', 'noise']
                random_type = random.choice(signal_types)
                
                # Professional random parameters
                params = {
                    'amplitude': random.uniform(0.5, 2.0),
                    'length': random.randint(500, 2000)
                }
                
                if random_type == 'sine':
                    params.update({
                        'frequency': random.uniform(1, 20),
                        'phase': random.uniform(-np.pi, np.pi)
                    })
                
                simulator = InteractiveSignalSimulator()
                samples = simulator.generate_signal(random_type, params)
                st.session_state.current_samples = samples
                st.session_state.current_signal_type = random_type
                st.success(f"Generated random {random_type} test signal")
                st.rerun()
        
        # Professional system controls
        st.markdown("---")
        
        if st.button("🗑️ Clear Laboratory Data", help="Reset all generated signals and recordings"):
            for key in ['signal_history', 'current_samples', 'voice_raw', 'voice_processed']:
                if 'samples' in key or 'voice' in key:
                    st.session_state[key] = np.array([])
                else:
                    st.session_state[key] = []
            st.success("✓ Laboratory data cleared")
            st.rerun()
        
        # Professional session metrics
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        
        signals_generated = len(st.session_state.signal_history)
        current_samples = len(st.session_state.current_samples)
        has_voice_recording = len(st.session_state.voice_raw) > 0
        
        st.metric("Generated Signals", f"{signals_generated}")
        st.metric("Active Sample Count", f"{current_samples:,}")
        st.metric("Voice Analysis", "Active" if has_voice_recording else "Standby")
        
        # Professional resources
        st.markdown("---")
        st.markdown("### 📖 Professional Resources")
        
        with st.expander("🧮 Mathematical References"):
            st.markdown("""
            **Core Transform Operations:**
            - Amplitude Scaling: `y(n) = k × x(n)`
            - Amplitude Shift: `y(n) = x(n) + c`
            - Time Scaling: `y(n) = x(αn)`
            - Time Shift: `y(n) = x(n-k)`
            """)
        
        with st.expander("🎯 Learning Objectives"):
            st.markdown("""
            **Professional Competencies:**
            - Discrete-time signal theory and analysis
            - Linear time-invariant system characterization
            - Digital signal processing operations
            - Professional audio signal analysis
            - Statistical signal characterization
            """)
    
    # Professional main content dispatch
    if page == "📊 Signal Generation Laboratory":
        signal_generator_page()
    elif page == "🎤 Voice Analysis Laboratory":
        voice_dts_processor_page()
    
    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #4a5568; padding: 20px; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 8px; margin-top: 2rem;">
        <h3 style="color: #2d3748; font-family: 'Inter', sans-serif; font-weight: 600;">Professional DTS Analysis Platform</h3>
        <p style="margin: 0.5rem 0;">Advanced discrete-time signal processing for engineering and research applications</p>
        <p style="margin: 0; font-size: 0.9rem;"><strong>Domain Expertise:</strong> Digital Signal Processing • Linear Systems Theory • Audio Engineering • Research Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
