import gradio as gr
import numpy as np
import joblib
import os

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = joblib.load(model_path)

# Try to load scaler, if not exists, create a dummy one
try:
    scaler = joblib.load(scaler_path)
except:
    scaler = None

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    """Predict diabetes based on patient health metrics."""
    try:
        # Create BMI category
        if bmi < 18.5:
            bmi_cat = 0
        elif bmi < 25:
            bmi_cat = 1
        elif bmi < 30:
            bmi_cat = 2
        else:
            bmi_cat = 3
        
        # Create Age group
        if age < 30:
            age_grp = 0
        elif age < 45:
            age_grp = 1
        elif age < 60:
            age_grp = 2
        else:
            age_grp = 3
        
        # Prepare features (10 features total)
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age, bmi_cat, age_grp]])
        
        # Scale features if scaler exists
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Format result
        if prediction == 1:
            result = "⚠️ **DIABETIC**"
            confidence = probability[1] * 100
        else:
            result = "✅ **NOT DIABETIC**"
            confidence = probability[0] * 100
        
        return f"{result}\n\n**Confidence:** {confidence:.1f}%"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface with separate inputs and sliders
with gr.Blocks(theme=gr.themes.Soft(), title="Diabetes Prediction System") as demo:
    gr.Markdown("# 🏥 Diabetes Prediction System")
    gr.Markdown("Enter patient health metrics below to predict diabetes risk using **Voting Classifier** (Best Model: 77.69% accuracy)")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Information")
            pregnancies = gr.Slider(
                label="Number of Pregnancies",
                minimum=0, maximum=20, step=1, value=1,
                info="Number of times pregnant (0-20)"
            )
            age = gr.Slider(
                label="Age (years)",
                minimum=18, maximum=100, step=1, value=30,
                info="Patient age in years (18-100)"
            )
            
        with gr.Column():
            gr.Markdown("### Blood Metrics")
            glucose = gr.Slider(
                label="Glucose Level (mg/dL)",
                minimum=50, maximum=250, step=1, value=120,
                info="Plasma glucose concentration (50-250)"
            )
            blood_pressure = gr.Slider(
                label="Blood Pressure (mm Hg)",
                minimum=40, maximum=140, step=1, value=70,
                info="Diastolic blood pressure (40-140)"
            )
            insulin = gr.Slider(
                label="Insulin (μU/ml)",
                minimum=0, maximum=500, step=1, value=80,
                info="2-Hour serum insulin (0-500)"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Body Measurements")
            bmi = gr.Slider(
                label="BMI (kg/m²)",
                minimum=15, maximum=55, step=0.1, value=25.0,
                info="Body Mass Index (15-55)"
            )
            skin_thickness = gr.Slider(
                label="Skin Thickness (mm)",
                minimum=0, maximum=80, step=1, value=20,
                info="Triceps skin fold thickness (0-80)"
            )
            
        with gr.Column():
            gr.Markdown("### Genetic Factor")
            dpf = gr.Slider(
                label="Diabetes Pedigree Function",
                minimum=0.05, maximum=2.5, step=0.01, value=0.5,
                info="Diabetes heredity score (0.05-2.5)"
            )
    
    with gr.Row():
        submit_btn = gr.Button("🔍 Predict Diabetes Risk", variant="primary", size="lg")
        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
    
    output = gr.Markdown(label="Prediction Result", value="*Click 'Predict' to see results*")
    
    # Submit button action
    submit_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
        outputs=output
    )
    
    # Clear button action
    def clear_inputs():
        return 1, 120, 70, 20, 80, 25.0, 0.5, 30, "*Click 'Predict' to see results*"
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 📋 Quick Examples")
    
    with gr.Row():
        ex1_btn = gr.Button("Example 1: High Risk", size="sm")
        ex2_btn = gr.Button("Example 2: Low Risk", size="sm")
        ex3_btn = gr.Button("Example 3: Moderate", size="sm")
    
    def load_example_1():
        return 6, 148, 72, 35, 0, 33.6, 0.627, 50
    
    def load_example_2():
        return 1, 85, 66, 29, 0, 26.6, 0.351, 31
    
    def load_example_3():
        return 3, 120, 70, 25, 100, 28.0, 0.45, 40
    
    ex1_btn.click(fn=load_example_1, outputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    ex2_btn.click(fn=load_example_2, outputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    ex3_btn.click(fn=load_example_3, outputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    
    gr.Markdown("---")
    gr.Markdown("*Model: Voting Classifier (Logistic Regression + Random Forest + Gradient Boosting)*")

if __name__ == "__main__":
    demo.launch()
