
# EEG Motor Imagery Dataset - Organized Structure

## Dataset Information
- Source: "An EEG motor imagery dataset for brain computer interface in acute stroke patients"
- Total subjects: 100
- Data formats: .mat (raw) and .edf (preprocessed)
- Sampling rate: 500 Hz
- Channels: 30 EEG + 2 EOG + 1 marker (based on paper)
- Trials per subject: 40 (20 left hand, 20 right hand MI)



## File Naming Convention
- Raw files: [subject_id]_task-motor-imagery_eeg.mat
- Preprocessed files: [subject_id]_task-motor-imagery_eeg.edf
- Example: sub-01_task-motor-imagery_eeg.mat (raw)
- Example: sub-01_task-motor-imagery_eeg.edf (preprocessed)

## Patient Information
- Total: 50
- Age range: 31 - 77 years
- Gender: {'male': np.int64(39), 'female': np.int64(11)}

## Analysis Notes
1. All EEG files have been verified to contain C3 and C4 channels
2. Paper's preprocessed data: 0.5-40 Hz filtered
3. For MI analysis, apply additional 8-30 Hz filter as per paper
4. Event markers should be used to segment 4-second MI periods
5. Raw data (.mat) contains trial labels: 1=left hand MI, 2=right hand MI

## Created on
2025-12-16 11:14:48

