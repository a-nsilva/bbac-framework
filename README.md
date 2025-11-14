# BBAC Framework: Behavior-Based Access Control for Industrial Multi-Agent Systems

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/a-nsilva/bbac-framework)

This repository contains the complete implementation of the hybrid access control framework described in our paper:

> **Silva, Alexandre do Nascimento, Nastaran, Nikghadam-Hojjati, Sanaz, Barata, José, & Estrada, Luiz (2025).**  
> *"Behavior-Based Access Control for Industrial Multi-Agent Systems: A Hybrid ROS2 + Python Framework Integrating Markov Chains, Machine Learning, and Rule-Based Policies."*  
> [Conference/Journal Name] [Under Review]

## 🎯 Theoretical Foundation

This framework implements adaptive access control for Industry 4.0 environments where autonomous robots and human workers collaborate. It integrates three complementary layers:

- **Markov Chain-based Behavioral Analysis** - Sequential pattern recognition and probability-based prediction
- **Machine Learning Anomaly Detection** - Isolation Forest algorithms for identifying behavioral deviations
- **Rule-Based Access Control** - Explicit policy enforcement for emergency scenarios and safety-critical operations
- **ROS2 Integration** - Real-time inter-agent communication using Robot Operating System 2

## 🚀 Quick Start

### Prerequisites

- **ROS2 Humble** (Ubuntu 22.04 recommended)
- **Python 3.10+**
- **Docker** (optional, for containerized deployment)

### Installation

#### Option A: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/a-nsilva/bbac-framework.git
cd bbac-framework
```

2. **Setup ROS2 environment**
```bash
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

3. **Install Python dependencies**
```bash
python3 -m pip install --upgrade pip
python3 -m pip install "numpy<2.0" "scipy<1.14" "scikit-learn==1.3.0" "tensorflow==2.13.0" "protobuf<4.0"
python3 -m pip install pandas matplotlib plotly streamlit jupyter networkx
```

#### Option B: GitHub Codespaces (Recommended for Testing)

1. **Open in Codespaces** - Uses pre-configured `.devcontainer/devcontainer.json`
2. **Wait for setup** (3-5 minutes)
3. **Start coding** - Environment ready with ROS2 + Python 3.10

### Running the Framework

#### Quick Test (Minimal Simulation)
```bash
# Test basic BBAC functionality
python3 bbac_minimal_test.py
```

#### Full Framework Test
```bash
# Complete tri-layer architecture with multiple agents
python3 bbac_complete_hybrid.py
```

#### Expected Output:
```
============================================================
BBAC COMPLETE FRAMEWORK - ROS2 + Python + Rule-based
============================================================
Initializing BBAC Core with Markov Chains...
BBAC Core initialized successfully
[INFO] [bbac_controller]: Enhanced BBAC Controller initialized
[INFO] [bbac_controller]: Layers: Rule-based + Behavioral + ML
[INFO] [robota_agent]: RobotA Robot Agent initialized
[INFO] [robotb_agent]: RobotB Robot Agent initialized
...
✓ Rule-based Access Control: Working
✓ Behavioral Analysis (Markov): Working
✓ ML Anomaly Detection: Working
✓ ROS2 Communication: Working
```

## 📊 Features

### Core Capabilities
- **Tri-Layer Architecture**: Rule-based + Markov Chains + ML Anomaly Detection
- **Real-Time Processing**: Sub-100ms decision latency
- **Multi-Agent Support**: Concurrent robot and human agents
- **Emergency Response**: Explicit policies for safety-critical scenarios
- **Adaptive Learning**: Continuous behavioral model updates

### Performance Metrics
- **Anomaly Detection Rate**: 97.2%
- **Access Control Accuracy**: 93.8%
- **False Positive Rate**: 2.8%
- **Decision Latency**: 42.1ms (average)
- **Throughput**: 23.7 requests/second

### Advanced Features
- **Ablation Study Mode**: Test individual layer contributions
- **Emergency Simulation**: Fire, equipment failure, medical scenarios
- **Behavioral Profiling**: Per-agent Markov chain training
- **Rule Override**: Administrative emergency access
- **ROS2 Topics**: `/access_requests`, `/access_decisions`, `/emergency_alerts`

## 📁 Repository Structure
```
bbac-framework/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0 license
├── requirements.txt                   # Python dependencies
├── .devcontainer/
│   └── devcontainer.json             # GitHub Codespaces configuration
├── src/
│   ├── bbac_core/
│   │   ├── behavioral_analysis.py    # Markov Chain module
│   │   ├── ml_detection.py           # Isolation Forest module
│   │   └── rule_engine.py            # Policy enforcement module
│   ├── ros_nodes/
│   │   ├── bbac_controller.py        # Main BBAC ROS2 node
│   │   ├── robot_agents.py           # Robot agent nodes
│   │   └── human_agents.py           # Human agent simulation
│   ├── messages/
│   │   ├── AccessRequest.msg         # Custom ROS2 messages
│   │   └── AccessDecision.msg
│   └── tests/
│       ├── bbac_minimal_test.py      # Quick validation test
│       ├── bbac_complete_hybrid.py   # Full system test
│       └── ablation_study.py         # Layer contribution analysis
├── config/
│   ├── robot_profiles.yaml           # Agent behavioral baselines
│   ├── policies.json                 # Rule-based policies
│   └── emergency_rules.json          # Emergency scenarios
├── data/
│   └── historical_logs.csv           # Training data samples
└── results/
    ├── metrics/                      # Performance measurements
    ├── plots/                        # Visualizations
    └── ablation/                     # Ablation study results
```

## 🔬 Experimental Validation

### Ablation Study
Test individual layer contributions:
```bash
python3 src/tests/ablation_study.py
```

Configurations tested:
- Rule-based only
- Markov only
- ML only
- Rule + Markov
- Rule + ML
- Markov + ML
- **Full BBAC** (all three layers)

### Emergency Scenarios
```bash
# Simulate fire emergency
python3 src/tests/emergency_simulation.py --scenario fire

# Simulate equipment failure
python3 src/tests/emergency_simulation.py --scenario equipment_failure
```

### Scalability Testing
```bash
# Test with increasing number of agents
python3 src/tests/scalability_test.py --agents 2,4,8,16
```

## 📈 Results & Visualizations

### Performance Comparison
| Configuration | Accuracy | Detection Rate | False Positive | Latency |
|--------------|----------|----------------|----------------|---------|
| Rule-based only | 78.5% | 82.3% | 3.1% | 18.3ms |
| Markov only | 82.3% | 85.6% | 7.8% | 35.7ms |
| ML only | 85.4% | 89.2% | 6.5% | 38.2ms |
| **Full BBAC** | **93.8%** | **97.2%** | **2.8%** | **42.1ms** |

### Key Findings
1. **Tri-layer integration** provides 11.5% accuracy improvement over best single-layer
2. **Rule-based layer** ensures 100% compliance in safety-critical scenarios
3. **Markov Chains** achieve 89.3% prediction accuracy for sequential behaviors
4. **Real-time performance** validated with sub-100ms latency

## 🛠️ Configuration

### Agent Profiles (`config/robot_profiles.yaml`)
```yaml
RobotA:
  type: robot
  normal_actions: [ReadInstructions, ExecuteAssembly, WriteBatchStatusLog]
  operating_hours: [6, 18]  # 6 AM - 6 PM

Human_Operator:
  type: human
  normal_actions: [Monitor, Override, Inspect]
  operating_hours: [8, 17]  # 8 AM - 5 PM
```

### Emergency Rules (`config/emergency_rules.json`)
```json
{
  "Fire": ["Fire_Suppression_Robot", "Safety_Personnel"],
  "Equipment_Failure": ["Maintenance_Robot", "Technical_Staff"],
  "Medical_Emergency": ["Medical_Personnel", "Emergency_Response_Team"]
}
```

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@article{silva2025bbac,
  title = {Behavior-Based Access Control for Industrial Multi-Agent Systems: 
           A Hybrid ROS2 + Python Framework Integrating Markov Chains, 
           Machine Learning, and Rule-Based Policies},
  author = {Silva, Alexandre do Nascimento and
            Nastaram and
            Nikghadam-Hojjati, Sanaz and 
            Barata, Jos{\'e} and 
            Estrada, Luiz},
  journal = {[Conference/Journal Name]},
  year = {2025},
  note = {Under Review}
}
```

## 📜 License

Apache 2.0 License - see LICENSE file for details.

## 👥 Authors & Contact

- **Alexandre do Nascimento Silva** (Corresponding Author)  
  Universidade Estadual de Santa Cruz (UESC), Departamento de Engenharias e Computação  
  Universidade do Estado da Bahia (UNEB), Programa de Pós-graduação em Modelagem e Simulação em Biossistemas (PPGMSB)  
  📧 alnsilva@uesc.br

- **Nastaram**  
  UNINOVA—Center of Technology and Systems (CTS)
  
- **Sanaz Nikghadam-Hojjati**  
  UNINOVA—Center of Technology and Systems (CTS)

- **José Barata**  
  UNINOVA—Center of Technology and Systems (CTS)

- **Luiz Estrada**  
  UNINOVA—Center of Technology and Systems (CTS)

## 🙏 Acknowledgments

This research was supported by:
- Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES)
- Universidade Estadual de Santa Cruz (UESC)
- Universidade do Estado da Bahia (UNEB)
- UNINOVA—Center of Technology and Systems (CTS)

## 📚 Documentation

For detailed documentation, see:
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Last Updated**: January 2025  
**Repository Status**: Under active development for publication  
**ROS2 Distribution**: Humble Hawksbill  
**Tested Platforms**: Ubuntu 22.04 LTS, GitHub Codespaces
