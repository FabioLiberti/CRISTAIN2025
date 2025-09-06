```markdown
# FA-FedAvg: Criticality-Aware Federated Learning with Cryptographic Verification and Blockchain-Verified Evidence Integrity for Inter-Agency Crime Intelligence

## Abstract

This repository contains the implementation of FA-FedAvg (Forensic-Aware Federated Averaging), a novel federated learning framework specifically designed for secure inter-agency collaboration in Italian law enforcement. The framework addresses the critical challenge of enabling collaborative intelligence analysis across Italy's fragmented law enforcement ecosystem—comprising Carabinieri (112,000 officers), Polizia di Stato (98,000), Guardia di Finanza (60,000), and 8,000 local stations—without compromising data sovereignty or operational security.

**Primary Contributions:**
- Criticality-aware weighted gradient aggregation based on crime severity (Art. 416-bis: weight 1.0, Art. 624: weight 0.35)
- Blockchain-based cryptographic verification ensuring gradient integrity and legal admissibility
- Hierarchical architecture reducing communication complexity from O(Kd) to O(√Kd)
- Differential privacy guarantees with (ε=0.5, δ=10⁻⁷) per round

## System Requirements

### Hardware Requirements
- CPU: Intel Xeon E5-2690 v4 or equivalent
- RAM: 32GB minimum (64GB recommended for full-scale simulations)
- Storage: 200GB for datasets and blockchain storage
- Network: 100 Mbps minimum bandwidth per node
- GPU: NVIDIA Tesla V100 (optional, for accelerated training)

### Software Requirements
- Operating System: Ubuntu 20.04 LTS or CentOS 8
- Python 3.8.10 or higher
- PyTorch 1.13.0
- CUDA 11.6 (for GPU support)
- OpenSSL 1.1.1 (for cryptographic operations)

## Installation

### Repository Setup

```bash
git clone https://github.com/FabioLiberti/CRISTAIN2025.git
cd CRISTAIN2025
```

### Environment Configuration

```bash
# Create isolated Python environment
python3 -m venv fa_fedavg_env
source fa_fedavg_env/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "from cryptography.hazmat.primitives import hashes; print('Cryptography module: OK')"
```

### Data Preparation

```bash
# Generate ISTAT-calibrated synthetic crime data
python data/generate_synthetic_italian_crime.py \
    --num_records 1500000 \
    --distribution config/istat_crime_distribution_2019_2023.json \
    --output data/synthetic_italian_crime.csv

# Validate synthetic data against ISTAT statistics
python scripts/validate_synthetic_data.py \
    --synthetic data/synthetic_italian_crime.csv \
    --reference data/istat_statistics.csv
```

## Core Implementation

### Algorithm Architecture

The FA-FedAvg framework implements a three-tier hierarchical architecture mapping to Italian law enforcement structure:

```python
from fa_fedavg.core import FAFedAvgServer, FAFedAvgClient
from fa_fedavg.privacy import AdaptiveGaussianMechanism
from fa_fedavg.blockchain import PBFTConsensus

# Initialize central coordinator (Ministry of Interior level)
server = FAFedAvgServer(
    num_rounds=100,
    num_clients=106,  # Provincial commands
    privacy_budget=(0.5, 1e-7),
    consensus_protocol='PBFT',
    aggregation_strategy='hierarchical'
)

# Configure criticality matrix based on Italian Penal Code
criticality_matrix = {
    'art_416_bis': 1.00,  # Mafia association
    'art_648_bis': 0.78,  # Money laundering
    'art_575_580': 0.95,  # Homicide, serious injury
    'art_624_625': 0.42,  # Aggravated theft
    'art_73_dpr_309': 0.30,  # Minor drug offenses
    'contraventions': 0.10  # Administrative violations
}

server.set_criticality_weights(criticality_matrix)
```

### Client-Side Training

```python
# Initialize provincial client (e.g., Questura di Roma)
client = FAFedAvgClient(
    client_id='questura_roma',
    local_dataset=local_crime_data,
    model=crime_detection_model,
    criticality_aware=True
)

# Perform local weighted training
for epoch in range(local_epochs):
    for batch in client.get_weighted_batches():
        # Compute gradients weighted by crime criticality
        weighted_gradients = client.compute_weighted_gradients(
            batch, 
            criticality_matrix
        )
        
        # Add differential privacy noise
        private_gradients = privacy_mechanism.add_noise(
            weighted_gradients,
            sensitivity=2.0 * max(criticality_matrix.values()),
            epsilon=0.5
        )
        
        # Generate cryptographic proof
        gradient_hash = hashlib.sha256(
            private_gradients.tobytes()
        ).hexdigest()
        
        # Submit to blockchain
        tx_hash = blockchain.submit_gradient_proof(
            gradient_hash,
            client_id,
            round_number
        )
```

## Experimental Reproduction

### Dataset Configuration

The framework uses synthetic data calibrated on ISTAT crime statistics (2019-2023):

```python
# Crime distribution based on Italian national statistics
crime_distribution = {
    'organized_crime': 0.02,  # 2% - Art. 416-bis and related
    'financial_crimes': 0.08,  # 8% - Art. 648-bis/ter
    'property_crimes': 0.40,  # 40% - Art. 624-625
    'drug_offenses': 0.20,  # 20% - Art. 73 DPR 309/90
    'other_crimes': 0.30  # 30% - Various
}

# Temporal patterns from ISTAT analysis
temporal_patterns = {
    'weekend_surge': 1.32,  # 32% increase in weekend crimes
    'summer_peak': 1.15,  # 15% increase in summer months
    'night_concentration': 0.65  # 65% of violent crimes at night
}
```

### Running Experiments

#### Baseline Comparison

```bash
# Compare FA-FedAvg with standard federated learning approaches
python experiments/baseline_comparison.py \
    --methods "fedavg,fedprox,qfedavg,dpfedavg,fafedavg" \
    --dataset synthetic_italian_crime \
    --num_clients 106 \
    --num_rounds 100 \
    --output results/baseline_comparison.json
```

#### Ablation Study

```bash
# Analyze contribution of each component
python experiments/ablation_study.py \
    --components "weighted_gradients,adaptive_noise,hierarchical_aggregation,blockchain" \
    --dataset synthetic_italian_crime \
    --metrics "accuracy,privacy,latency,communication" \
    --output results/ablation_analysis.json
```

#### Scalability Analysis

```bash
# Test scalability up to 8000 clients (full Italian deployment)
python experiments/scalability_test.py \
    --min_clients 10 \
    --max_clients 8000 \
    --step_size 100 \
    --measure "convergence_time,communication_overhead,memory_usage" \
    --output results/scalability_metrics.json
```

## Evaluation Metrics

### Performance Metrics

| Metric | FA-FedAvg | FedAvg | DP-FedAvg | q-FedAvg |
|--------|-----------|---------|-----------|----------|
| Accuracy | 91.2±0.4% | 89.3±0.5% | 87.1±0.7% | 88.1±0.6% |
| F1-Score (Organized Crime) | 0.893 | 0.856 | 0.821 | 0.839 |
| F1-Score (Financial Crime) | 0.876 | 0.842 | 0.808 | 0.825 |
| Convergence Rounds | 100 | 100 | 100 | 110 |
| Communication per Round | O(√Kd) | O(d) | O(d) | O(d) |

### Privacy Guarantees

| Privacy Level | Per-round ε | Total ε (100 rounds) | Accuracy Impact |
|---------------|-------------|---------------------|-----------------|
| No Privacy | ∞ | ∞ | 92.1% |
| Low Privacy | 2.0 | ~20 | 91.8% |
| Medium Privacy | 1.0 | ~10 | 91.5% |
| High Privacy (FA-FedAvg) | 0.5 | ~5 | 91.2% |
| Very High Privacy | 0.1 | ~1 | 88.7% |

### Security Analysis

| Attack Type | Success Rate | Defense Mechanism |
|-------------|--------------|-------------------|
| Membership Inference | 51.3% (near random) | Differential Privacy (ε=0.5) |
| Model Poisoning | <5% with 30% malicious | Byzantine-robust aggregation |
| Gradient Manipulation | 0% (detected) | Blockchain verification |
| Data Reconstruction | Negligible | Secure aggregation + DP |

## Theoretical Foundations

### Convergence Guarantee

**Theorem 1 (Weighted Convergence):** Under L-smoothness and bounded gradient assumptions, FA-FedAvg converges at rate:

```
E[F(θ_T)] - F(θ*) ≤ L||θ_0 - θ*||²/(T·η_T) + η·L·(σ² + G²·σ²_noise)/K
```

where optimal learning rate η_t = 1/√(KT).

### Privacy Composition

**Theorem 2 (Privacy Budget):** FA-FedAvg satisfies (ε_total, δ_total)-differential privacy where:
- Per-round: (ε_r = 0.5, δ_r = 10⁻⁷)
- After T=100 rounds: ε_total ≈ 5 under strong composition
- With Rényi DP: ε_total ≈ 3.8 (tighter bound)

## Deployment Architecture

### Hierarchical Structure

```
Level 0: National Coordinator (Ministry of Interior)
├── Level 1: Regional Aggregators (20 regions)
│   ├── Level 2: Provincial Commands (106 provinces)
│   │   └── Level 3: Local Stations (8,000 stations)
│   └── Blockchain Verification Layer (PBFT Consensus)
└── Europol/Interpol Gateway (International Cooperation)
```

### Communication Protocol

```python
# Inter-agency secure communication protocol
class InterAgencyProtocol:
    def __init__(self):
        self.tls_config = {
            'version': 'TLSv1.3',
            'cipher_suite': 'AES256-GCM-SHA384',
            'mutual_auth': True,
            'cert_validation': 'strict'
        }
        
    def exchange_gradients(self, sender_agency, receiver_agency, gradients):
        # 1. Mutual TLS authentication
        session = self.establish_secure_channel(sender_agency, receiver_agency)
        
        # 2. Serialize with Protocol Buffers
        serialized = gradients.SerializeToString()
        
        # 3. Encrypt and sign
        encrypted = session.encrypt(serialized)
        signature = sender_agency.sign(hashlib.sha256(serialized).digest())
        
        # 4. Log to blockchain
        tx_hash = blockchain.log_exchange(
            sender=sender_agency.id,
            receiver=receiver_agency.id,
            gradient_hash=hashlib.sha256(serialized).hexdigest(),
            timestamp=datetime.utcnow()
        )
        
        return encrypted, signature, tx_hash
```

## Blockchain Integration

### PBFT Consensus Implementation

```python
class PBFTConsensus:
    def __init__(self, num_nodes, fault_tolerance=0.33):
        self.num_nodes = num_nodes
        self.f = int(num_nodes * fault_tolerance)  # Byzantine nodes
        self.required_votes = 2 * self.f + 1
        
    def verify_gradient_update(self, gradient_hash, round_number):
        # Phase 1: Pre-prepare
        pre_prepare_msg = self.create_pre_prepare(gradient_hash, round_number)
        
        # Phase 2: Prepare
        prepare_votes = self.collect_prepare_votes(pre_prepare_msg)
        if len(prepare_votes) < self.required_votes:
            return False, "Insufficient prepare votes"
            
        # Phase 3: Commit
        commit_votes = self.collect_commit_votes(prepare_votes)
        if len(commit_votes) < self.required_votes:
            return False, "Insufficient commit votes"
            
        # Add to blockchain
        block = self.create_block(gradient_hash, round_number, commit_votes)
        return True, block.hash
```

## Testing Framework

### Unit Tests

```bash
# Run comprehensive test suite
pytest tests/ -v --cov=fa_fedavg --cov-report=html

# Test specific components
pytest tests/test_privacy_mechanism.py -v
pytest tests/test_blockchain_verification.py -v
pytest tests/test_hierarchical_aggregation.py -v
```

### Integration Tests

```bash
# Test multi-agency collaboration scenario
python tests/integration/test_multi_agency.py \
    --agencies "carabinieri,polizia_stato,guardia_finanza" \
    --rounds 50 \
    --verify_blockchain True
```

### Performance Benchmarks

```bash
# Benchmark communication efficiency
python benchmarks/communication_benchmark.py \
    --clients [10,50,100,500,1000,5000,8000] \
    --model_size 10000000 \
    --output benchmarks/communication_results.csv

# Benchmark convergence speed
python benchmarks/convergence_benchmark.py \
    --dataset synthetic_italian_crime \
    --algorithms "fafedavg,fedavg,fedprox" \
    --output benchmarks/convergence_results.csv
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{liberti2025fafedavg,
  title={Criticality-Aware Federated Learning with Cryptographic Verification 
         and Blockchain-Verified Evidence Integrity for Inter-Agency Crime Intelligence},
  author={Liberti, Fabio},
  booktitle={1st Workshop on Supporting Crime Resolution Through Artificial Intelligence (CRISTAIN)},
  pages={1--13},
  year={2025},
  organization={CHItaly 2025},
  address={Salerno, Italy},
  doi={10.1145/cristain.2025.liberti}
}
```

## License

This project is licensed under a personal license. See [LICENSE](LICENSE) file for details. All rights are reserved by the authors. This project may not be copied, distributed, or modified without explicit written permission.

## Disclaimer

This implementation is intended for research purposes within the academic and law enforcement research communities. The framework is designed to operate within existing legal and regulatory frameworks without requiring legislative modifications. Any production deployment must undergo appropriate security audits and regulatory compliance verification.

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback and the Italian Ministry of University and Research for supporting this work. Special acknowledgment to ISTAT for providing access to anonymized crime statistics used in calibrating our synthetic data generator.

## Contact

**Principal Investigator:** Fabio Liberti  
**Affiliation:** Department of Engineering and Science, University of Mercatorum  
**Email:** fabio.liberti@studenti.unimercatorum.it  
**ORCID:** 0000-0003-3019-5411  

For technical inquiries regarding the implementation, please open an issue in this repository.
```