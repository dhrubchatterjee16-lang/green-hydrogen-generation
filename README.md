# Green Hydrogen Generation: Electrochemical & Dynamic Multiphysics PEM Electrolyzer Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Physics: Butler-Volmer](https://img.shields.io/badge/Model-Nernst--Planck%20%26%20Butler--Volmer-emerald)]()

A high-fidelity dynamic multiphysics modeling engine for Proton Exchange Membrane (PEM) water electrolyzers operating under transient renewable energy input profiles.

---

## ⚡ Mathematical & Electrochemical Foundations

The cell polarization voltage $V_{\text{cell}}$ is modeled from first-principles electrochemistry across activation, ohmic, and concentration overpotentials:

$$V_{\text{cell}}(t) = E_{\text{rev}}(T, P) + \eta_{\text{act,anode}} + \eta_{\text{act,cathode}} + \eta_{\text{ohmic}} + \eta_{\text{diff}}$$

1. **Thermodynamic Reversible Potential ($E_{\text{rev}}$):**
   $$E_{\text{rev}} = E_0 + \frac{RT}{2F} \ln\left(\frac{P_{\text{H}_2} \cdot P_{\text{O}_2}^{0.5}}{a_{\text{H}_2\text{O}}}\right)$$
2. **Activation Overpotentials (Butler-Volmer Kinetics):**
   $$j = j_0 \left[ \exp\left(\frac{\alpha_a F \eta_{\text{act}}}{RT}\right) - \exp\left(-\frac{\alpha_c F \eta_{\text{act}}}{RT}\right) \right]$$
3. **Membrane Ohmic Resistance (Nafion Hydration & Conductivity):**
   $$\sigma_{\text{mem}}(\lambda, T) = (0.005139\lambda - 0.00326) \exp\left(1268 \left(\frac{1}{303} - \frac{1}{T}\right)\right)$$
4. **Thermal Transient Dynamics:**
   $$C_{\text{th}} \frac{dT_{\text{stack}}}{dt} = \dot{Q}_{\text{gen}} - \dot{Q}_{\text{cool}} - \dot{Q}_{\text{loss}}$$

---

## 📁 Repository Structure

```
green-hydrogen-generation/
├── src/
│   ├── electrochemistry/    # Butler-Volmer and Nernst potential solvers
│   ├── membrane/            # Nafion conductivity & water drag transport
│   ├── thermal/             # Stack lumped-capacitance heat balance
│   └── degradation/         # Catalyst dissolution & membrane thinning models
├── data/                    # Renewable solar & wind intermittent profiles
├── simulations/             # Polarization curves and dynamic step-response
├── requirements.txt         # NumPy, SciPy, Matplotlib
└── run_simulation.py        # Main execution entrypoint
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/dhrubchatterjee16-lang/green-hydrogen-generation.git
cd green-hydrogen-generation
pip install -r requirements.txt
python run_simulation.py --current_density 1.8 --temp 353 --pressure 30
```

*(For techno-economic LCOH optimization and capital allocation, see [GREEN_HYDROGEN_generation-](https://github.com/dhrubchatterjee16-lang/GREEN_HYDROGEN_generation-)).*
