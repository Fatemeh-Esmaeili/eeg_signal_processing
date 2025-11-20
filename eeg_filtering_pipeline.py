"""
EEG Data Filtering Pipeline
===========================

A professional pipeline for filtering EEG data with comprehensive visualization
and quality reporting.

Author: FE
Date: 2025
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mne import create_info
from mne.io import RawArray
from mne.filter import filter_data
from mne.preprocessing import ICA
import mne
from mne_bids import BIDSPath, read_raw_bids
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class EEGFilteringPipeline:
    """
    A comprehensive pipeline for EEG data filtering and quality assessment.
    
    Features:
    - Batch processing of EEG files
    - Comprehensive quality assessment
    - Professional visualizations
    - Filtering comparison plots
    - Quality control reports
    """
    
    def __init__(self, base_path, output_dir="processed_eeg"):
        """
        Initialize the EEG filtering pipeline.
        
        Parameters
        ----------
        base_path : str
            Path to the base directory containing EEG data
        output_dir : str
            Output directory for processed data and reports
        """
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.setup_logging()
        
        # Filtering parameters
        self.filter_params = {
            'l_freq': 1.0,
            'h_freq': 40.0,
            'method': 'fir',
            'fir_window': 'hamming',
            'phase': 'zero-double',
            'filter_length': 'auto'
        }
        
        # Visualization parameters
        self.viz_params = {
            'figsize': (15, 10),
            'dpi': 300,
            'fontsize': 12
        }
        
    def setup_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / "filtered_eeg",
            self.output_dir / "quality_reports",
            self.output_dir / "filtering_reports", 
            self.output_dir / "quality_visualizations",
            self.output_dir / "filtering_comparisons"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Set up professional logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EEG_Filtering_Pipeline')
        
    def load_eeg_files(self, file_pattern="**/*.vhdr"):
        """
        Load EEG files from the specified directory.
        
        Parameters
        ----------
        file_pattern : str
            Glob pattern for finding EEG files
            
        Returns
        -------
        list
            List of file paths
        """
        self.logger.info("Scanning for EEG files...")
        files = list(self.base_path.glob(file_pattern))
        self.logger.info(f"Found {len(files)} EEG files")
        return files
    
    def create_quality_report(self, raw, file_path):
        """
        Create comprehensive quality assessment report.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        file_path : Path
            Path to the original file
            
        Returns
        -------
        dict
            Quality metrics dictionary
        """
        self.logger.info(f"Creating quality report for {file_path.name}")
        
        # Calculate basic metrics
        duration = raw.times[-1]
        n_channels = len(raw.ch_names)
        sfreq = raw.info['sfreq']
        
        # Calculate channel-wise statistics
        data = raw.get_data()
        channel_metrics = {}
        
        for i, ch_name in enumerate(raw.ch_names):
            channel_data = data[i]
            channel_metrics[ch_name] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'variance': float(np.var(channel_data)),
                'max': float(np.max(channel_data)),
                'min': float(np.min(channel_data)),
                'range': float(np.ptp(channel_data))
            }
        
        # Global metrics
        global_metrics = {
            'file_name': file_path.name,
            'duration_seconds': float(duration),
            'sampling_frequency': float(sfreq),
            'n_channels': n_channels,
            'total_samples': data.shape[1],
            'data_shape': list(data.shape),
            'global_mean': float(np.mean(data)),
            'global_std': float(np.std(data)),
            'bad_channels': raw.info.get('bads', [])
        }
        
        # Combine metrics
        quality_report = {
            'global_metrics': global_metrics,
            'channel_metrics': channel_metrics,
            'file_info': {
                'original_path': str(file_path),
                'processing_date': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save report
        report_file = self.output_dir / "quality_reports" / f"{file_path.stem}_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2)
            
        return quality_report
    
    def create_quality_visualization(self, raw, file_path, quality_report):
        """
        Create professional quality assessment visualizations.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        file_path : Path
            Path to the original file
        quality_report : dict
            Quality metrics dictionary
        """
        self.logger.info(f"Creating quality visualization for {file_path.name}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Raw data overview
        ax1 = plt.subplot(2, 2, 1)
        data = raw.get_data()
        times = raw.times
        
        # Plot first few channels for clarity
        n_channels_plot = min(8, data.shape[0])
        for i in range(n_channels_plot):
            normalized_data = data[i] - np.mean(data[i])
            normalized_data = normalized_data / np.max(np.abs(normalized_data)) if np.max(np.abs(normalized_data)) > 0 else normalized_data
            plt.plot(times[:min(10000, len(times))], normalized_data[:min(10000, len(times))] + i, 
                    label=raw.ch_names[i], linewidth=0.8)
        
        ax1.set_title('EEG Data Overview (First 8 Channels)', fontsize=self.viz_params['fontsize'] + 2, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Amplitude + Offset')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Channel-wise standard deviation
        ax2 = plt.subplot(2, 2, 2)
        channel_stds = [quality_report['channel_metrics'][ch]['std'] for ch in raw.ch_names]
        channels = range(len(raw.ch_names))
        
        bars = ax2.bar(channels, channel_stds, alpha=0.7, color='skyblue')
        ax2.set_title('Channel-wise Standard Deviation', fontsize=self.viz_params['fontsize'] + 2, fontweight='bold')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Standard Deviation (µV)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Power spectral density
        ax3 = plt.subplot(2, 2, 3)
        try:
            # Compute PSD for first few channels
            spectrum = raw.compute_psd(fmax=50)
            psd, freqs = spectrum.get_data(return_freqs=True)
            
            for i in range(min(6, psd.shape[0])):
                ax3.semilogy(freqs, psd[i], label=raw.ch_names[i], alpha=0.7)
                
            ax3.set_title('Power Spectral Density', fontsize=self.viz_params['fontsize'] + 2, fontweight='bold')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power Spectral Density (dB/Hz)')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            self.logger.warning(f"PSD computation failed: {e}")
            ax3.text(0.5, 0.5, 'PSD Computation Failed', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Quality metrics summary
        ax4 = plt.subplot(2, 2, 4)
        metrics = quality_report['global_metrics']
        summary_text = (
            f"File: {metrics['file_name']}\n"
            f"Duration: {metrics['duration_seconds']:.1f}s\n"
            f"Sampling Rate: {metrics['sampling_frequency']:.1f} Hz\n"
            f"Channels: {metrics['n_channels']}\n"
            f"Samples: {metrics['total_samples']:,}\n"
            f"Global Mean: {metrics['global_mean']:.2f} µV\n"
            f"Global Std: {metrics['global_std']:.2f} µV\n"
            f"Bad Channels: {len(metrics['bad_channels'])}"
        )
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax4.set_title('Quality Metrics Summary', fontsize=self.viz_params['fontsize'] + 2, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        viz_file = self.output_dir / "quality_visualizations" / f"{file_path.stem}_quality_viz.png"
        plt.savefig(viz_file, dpi=self.viz_params['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Quality visualization saved: {viz_file}")
    
    def apply_filtering(self, raw, file_path):
        """
        Apply bandpass filtering to EEG data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        file_path : Path
            Path to the original file
            
        Returns
        -------
        mne.io.Raw
            Filtered EEG data
        """
        self.logger.info(f"Applying filtering to {file_path.name}")
        
        try:
            # Apply bandpass filter
            filtered_raw = raw.copy().filter(
                l_freq=self.filter_params['l_freq'],
                h_freq=self.filter_params['h_freq'],
                method=self.filter_params['method'],
                fir_window=self.filter_params['fir_window'],
                phase=self.filter_params['phase']
            )
            
            # Save filtered data
            filtered_file = self.output_dir / "filtered_eeg" / f"{file_path.stem}_filtered.fif"
            filtered_raw.save(filtered_file, overwrite=True)
            
            self.logger.info(f"Filtered data saved: {filtered_file}")
            return filtered_raw
            
        except Exception as e:
            self.logger.error(f"Filtering failed for {file_path.name}: {e}")
            return None
    
    def create_filtering_report(self, raw, filtered_raw, file_path):
        """
        Create filtering comparison report and visualization.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Original raw EEG data
        filtered_raw : mne.io.Raw
            Filtered EEG data
        file_path : Path
            Path to the original file
        """
        self.logger.info(f"Creating filtering report for {file_path.name}")
        
        if filtered_raw is None:
            self.logger.warning(f"No filtered data available for {file_path.name}")
            return
        
        # Calculate filtering metrics
        original_data = raw.get_data()
        filtered_data = filtered_raw.get_data()
        
        metrics = {
            'file_name': file_path.name,
            'original_std': float(np.std(original_data)),
            'filtered_std': float(np.std(filtered_data)),
            'noise_reduction_ratio': float(np.std(original_data) / np.std(filtered_data)),
            'filter_settings': self.filter_params,
            'processing_date': pd.Timestamp.now().isoformat()
        }
        
        # Save metrics
        report_file = self.output_dir / "filtering_reports" / f"{file_path.stem}_filtering_report.json"
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualization
        self.create_filtering_comparison_viz(raw, filtered_raw, file_path, metrics)
    
    def create_filtering_comparison_viz(self, raw, filtered_raw, file_path, metrics):
        """
        Create professional filtering comparison visualization.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Original raw EEG data
        filtered_raw : mne.io.Raw
            Filtered EEG data
        file_path : Path
            Path to the original file
        metrics : dict
            Filtering metrics dictionary
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Plot original vs filtered data for first few channels
        n_channels_plot = min(4, raw.info['nchan'])
        times = raw.times[:3000]  # First 3 seconds
        
        for i in range(n_channels_plot):
            plt.subplot(n_channels_plot, 1, i + 1)
            
            # Original data
            original_segment = raw.get_data(picks=[i])[0, :3000]
            plt.plot(times, original_segment, 'b-', alpha=0.7, linewidth=1, label='Original')
            
            # Filtered data
            filtered_segment = filtered_raw.get_data(picks=[i])[0, :3000]
            plt.plot(times, filtered_segment, 'r-', alpha=0.8, linewidth=1, label='Filtered')
            
            plt.ylabel(f'{raw.ch_names[i]}\n(µV)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            if i == 0:
                plt.title(f'Filtering Comparison: {file_path.stem}\n'
                         f'Bandpass: {self.filter_params["l_freq"]}-{self.filter_params["h_freq"]} Hz', 
                         fontsize=self.viz_params['fontsize'] + 2, fontweight='bold')
        
        plt.xlabel('Time (s)')
        plt.tight_layout()
        
        # Save comparison plot
        comp_file = self.output_dir / "filtering_comparisons" / f"{file_path.stem}_filtering_comparison.png"
        plt.savefig(comp_file, dpi=self.viz_params['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Filtering comparison saved: {comp_file}")
    
    def create_summary_comparison_plot(self, all_files):
        """
        Create a comprehensive summary comparison plot across all files.
        
        Parameters
        ----------
        all_files : list
            List of processed file paths
        """
        self.logger.info("Creating summary comparison plot")
        
        # Load quality metrics from all files
        quality_reports = []
        filtering_reports = []
        
        for file_path in all_files:
            # Load quality report
            quality_file = self.output_dir / "quality_reports" / f"{file_path.stem}_quality_report.json"
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    quality_reports.append(json.load(f))
            
            # Load filtering report
            filtering_file = self.output_dir / "filtering_reports" / f"{file_path.stem}_filtering_report.json"
            if filtering_file.exists():
                with open(filtering_file, 'r') as f:
                    filtering_reports.append(json.load(f))
        
        if not quality_reports or not filtering_reports:
            self.logger.warning("Insufficient data for summary comparison plot")
            return
        
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Global standard deviation comparison
        original_stds = [report['global_metrics']['global_std'] for report in quality_reports]
        filtered_stds = [report['filtered_std'] for report in filtering_reports]
        file_names = [report['file_name'] for report in quality_reports[:20]]  # First 20 for clarity
        
        x_pos = np.arange(len(original_stds[:20]))
        
        axes[0, 0].bar(x_pos - 0.2, original_stds[:20], 0.4, label='Original', alpha=0.7)
        axes[0, 0].bar(x_pos + 0.2, filtered_stds[:20], 0.4, label='Filtered', alpha=0.7)
        axes[0, 0].set_xlabel('File Index')
        axes[0, 0].set_ylabel('Global Standard Deviation (µV)')
        axes[0, 0].set_title('Noise Reduction Across Files', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Noise reduction ratio distribution
        noise_reduction = [report['noise_reduction_ratio'] for report in filtering_reports]
        axes[0, 1].hist(noise_reduction, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(noise_reduction), color='red', linestyle='--', label=f'Mean: {np.mean(noise_reduction):.2f}')
        axes[0, 1].set_xlabel('Noise Reduction Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Noise Reduction Ratio Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Data duration distribution
        durations = [report['global_metrics']['duration_seconds'] for report in quality_reports]
        axes[1, 0].hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.1f}s')
        axes[1, 0].set_xlabel('Duration (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('EEG Recording Duration Distribution', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Channel count distribution
        n_channels = [report['global_metrics']['n_channels'] for report in quality_reports]
        unique_channels, counts = np.unique(n_channels, return_counts=True)
        axes[1, 1].bar(unique_channels, counts, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Number of Channels')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Channel Count Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_file = self.output_dir / "filtering_comparison_summary.png"
        plt.savefig(summary_file, dpi=self.viz_params['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary comparison plot saved: {summary_file}")
    
    def run_pipeline(self, file_pattern="**/*.vhdr"):
        """
        Run the complete EEG filtering pipeline.
        
        Parameters
        ----------
        file_pattern : str
            Glob pattern for finding EEG files
        """
        self.logger.info("Starting EEG Filtering Pipeline")
        
        # Load files
        files = self.load_eeg_files(file_pattern)
        
        if not files:
            self.logger.error("No EEG files found!")
            return
        
        processed_files = []
        
        for i, file_path in enumerate(files):
            self.logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
            
            try:
                # Load EEG data
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
                
                # Create quality assessment
                quality_report = self.create_quality_report(raw, file_path)
                self.create_quality_visualization(raw, file_path, quality_report)
                
                # Apply filtering
                filtered_raw = self.apply_filtering(raw, file_path)
                
                # Create filtering report
                if filtered_raw is not None:
                    self.create_filtering_report(raw, filtered_raw, file_path)
                    processed_files.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                continue
        
        # Create summary comparison
        if processed_files:
            self.create_summary_comparison_plot(processed_files)
        
        # Generate final report
        self.generate_final_report(processed_files)
        
        self.logger.info("EEG Filtering Pipeline completed successfully!")
    
    def generate_final_report(self, processed_files):
        """Generate a final pipeline execution report."""
        report = {
            'pipeline_info': {
                'name': 'EEG Filtering Pipeline',
                'version': '1.0',
                'completion_time': pd.Timestamp.now().isoformat()
            },
            'processing_summary': {
                'total_files_found': len(self.load_eeg_files()),
                'files_processed': len(processed_files),
                'success_rate': len(processed_files) / len(self.load_eeg_files()) * 100,
                'filter_settings': self.filter_params
            },
            'output_structure': {
                'filtered_eeg': 'Filtered EEG data in FIF format',
                'quality_reports': 'JSON files with quality metrics',
                'filtering_reports': 'JSON files with filtering metrics',
                'quality_visualizations': 'PNG files with quality assessment plots',
                'filtering_comparisons': 'PNG files with filtering comparison plots'
            }
        }
        
        # Save final report
        report_file = self.output_dir / "pipeline_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Final report saved: {report_file}")


def main():
    """Main execution function."""
    # Configuration
    BASE_PATH = Path("/path/to/your/eeg/data")  # Update this path
    OUTPUT_DIR = "processed_eeg_output"
    
    # Initialize and run pipeline
    pipeline = EEGFilteringPipeline(BASE_PATH, OUTPUT_DIR)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()