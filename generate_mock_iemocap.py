import os
import random
import soundfile as sf
import numpy as np

def generate_mock_iemocap(root_dir="mock_iemocap"):
    """
    Generates a mock IEMOCAP dataset structure.
    Structure:
    SessionX/
        dialog/
            wav/
                Ses01F_script01_1_M001.wav
            EmoEvaluation/
                Ses01F_script01_1.txt
    """
    
    sessions = ['Session1', 'Session2']
    emotions = ['neu', 'hap', 'ang', 'sad', 'exc', 'fru', 'sur', 'fea', 'oth', 'xxx']
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    print(f"Generating mock data in {root_dir}...")
    
    for session in sessions:
        # Create directories
        wav_dir = os.path.join(root_dir, session, 'dialog', 'wav')
        emo_dir = os.path.join(root_dir, session, 'dialog', 'EmoEvaluation')
        
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(emo_dir, exist_ok=True)
        
        # Generate dummy dialogs
        for i in range(1, 3): # 2 dialogs per session
            dialog_id = f"Ses01F_script0{i}_1"
            
            # Create EmoEvaluation file
            emo_file_path = os.path.join(emo_dir, f"{dialog_id}.txt")
            with open(emo_file_path, 'w') as f:
                # Header lines (dummy)
                f.write(f"[%] {dialog_id}\n")
                f.write("Line 2\nLine 3\n") 
                
                # Generate turns
                for j in range(5): # 5 turns per dialog
                    turn_id = f"{dialog_id}_M{j:03d}"
                    emotion = random.choice(emotions)
                    # Format: [START_TIME - END_TIME] TURN_ID EMOTION [V, A, D]
                    line = f"[0.0000 - 1.0000]\t{turn_id}\t{emotion}\t[5.0000, 5.0000, 5.0000]\n"
                    f.write(line)
                    
                    # Create corresponding wav file
                    wav_path = os.path.join(wav_dir, f"{turn_id}.wav")
                    if not os.path.exists(wav_path):
                        # Generate 1 sec of white noise
                        sr = 16000 # IEMOCAP usually 16k
                        audio = np.random.uniform(-0.1, 0.1, sr)
                        sf.write(wav_path, audio, sr)
                        
    print("Mock data generation complete.")

if __name__ == "__main__":
    generate_mock_iemocap()
