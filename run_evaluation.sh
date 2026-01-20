#!/bin/bash
# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Phoenix V2 Evaluation Script
# Comprehensive evaluation comparing V1 vs V2

set -e

echo "========================================================================"
echo "PHOENIX V2 EVALUATION SUITE"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Run unit tests
echo -e "${BLUE}[1/4] Running Unit Tests...${NC}"
echo "------------------------------------------------------------------------"
python phoenix_v2/tests/test_phoenix_v2.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed${NC}"
fi
echo ""

# Step 2: Download datasets (if needed)
echo -e "${BLUE}[2/4] Checking Datasets...${NC}"
echo "------------------------------------------------------------------------"
if [ ! -d "./data/movielens-1m" ]; then
    echo "Downloading datasets (this may take a few minutes)..."
    python datasets/download_datasets.py
else
    echo -e "${GREEN}✓ Datasets already downloaded${NC}"
fi
echo ""

# Step 3: Run quick comparison
echo -e "${BLUE}[3/4] Running Quick Comparison (100 examples)...${NC}"
echo "------------------------------------------------------------------------"
python evaluation/run_comparison.py --num_examples 100 --emb_size 256 --num_layers 4
echo ""

# Step 4: Run ablation study
echo -e "${BLUE}[4/4] Running Ablation Study...${NC}"
echo "------------------------------------------------------------------------"
python evaluation/run_comparison.py --ablation --num_examples 200 --emb_size 256 --num_layers 4
echo ""

# Summary
echo "========================================================================"
echo -e "${GREEN}EVALUATION COMPLETE${NC}"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - ./evaluation_results/comparison_results.json"
echo "  - ./evaluation_results/ablation_study.json"
echo ""
echo "To view results:"
echo "  cat evaluation_results/comparison_results.json | jq '.'"
echo ""
echo "To run with more examples:"
echo "  python evaluation/run_comparison.py --num_examples 1000"
echo ""
