#!/bin/bash
# K-Fold十折叠交叉验证 - Bash启动脚本
# 用途: 快速启动K-Fold训练流程

set -e  # 出错时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "\n${GREEN}📌 $1${NC}"
    echo -e "${YELLOW}────────────────────────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 检查依赖
check_dependencies() {
    print_step "检查依赖"

    if ! command -v python3 &> /dev/null; then
        print_error "未找到 python3"
        exit 1
    fi
    print_success "Python: $(python3 --version)"

    if ! command -v nvidia-smi &> /dev/null; then
        print_info "NVIDIA GPU 不可用（将使用CPU训练）"
    else
        print_success "GPU 可用: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
}

# 检查文件
check_files() {
    print_step "检查项目文件"

    files=(
        "kfold_cross_validation.py"
        "kfold_data_loader.py"
        "kfold_shipsear_integration.py"
        "train_distillation_shipsear.py"
        "configs/train_distillation_shipsear.yaml"
    )

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            print_success "$file"
        else
            print_error "$file 不存在"
            exit 1
        fi
    done
}

# 生成K-Fold划分
generate_kfold() {
    print_step "生成K-Fold划分"

    read -p "请输入数据目录路径 (默认: ./data): " data_dir
    data_dir=${data_dir:-.\/data}

    if [ ! -d "$data_dir/train/wav" ]; then
        print_error "数据目录结构不对: $data_dir/train/wav 不存在"
        exit 1
    fi

    print_info "使用数据目录: $data_dir"
    print_info "生成K-Fold划分中..."

    python3 << 'EOF'
from kfold_cross_validation import KFoldCrossValidator
import os

data_dir = os.environ.get('KFOLD_DATA_DIR', './data')

validator = KFoldCrossValidator(
    data_dir=data_dir,
    output_dir='results/kfold_splits',
    seed=42
)

samples = validator.scan_dataset('train')
if not samples:
    print('❌ 无法加载数据')
    exit(1)

validator.generate_folds(samples)
validator.save_all_splits()
validator.save_split_indices()

print('✨ K-Fold划分生成完成!')
EOF

    if [ $? -ne 0 ]; then
        print_error "K-Fold划分生成失败"
        exit 1
    fi

    print_success "K-Fold划分已保存到: results/kfold_splits/"
}

# 验证划分
verify_splits() {
    print_step "验证划分结果"

    if [ ! -f "results/kfold_splits/kfold_summary.txt" ]; then
        print_error "未找到划分文件"
        exit 1
    fi

    print_info "显示前30行:"
    head -30 results/kfold_splits/kfold_summary.txt

    print_success "划分结果验证完成"
}

# 测试单个Fold
test_single_fold() {
    print_step "测试单个Fold"

    read -p "是否运行Fold 0的测试训练? (y/n, 默认n): " response
    response=${response:-n}

    if [ "$response" != "y" ]; then
        print_info "跳过单个Fold测试"
        return
    fi

    print_info "启动Fold 0的测试训练..."
    print_info "提示: 可以用 'nvidia-smi' 监控GPU使用"

    python3 kfold_shipsear_integration.py \
        --train-fold 0 \
        --gpus 4,5,6,7 \
        --splits-dir results/kfold_splits

    if [ $? -ne 0 ]; then
        print_error "Fold 0 测试失败"
    else
        print_success "Fold 0 测试成功"
    fi
}

# 运行所有Fold
run_all_folds() {
    print_step "批量训练所有Fold"

    read -p "是否运行所有Fold的训练? (y/n, 默认n): " response
    response=${response:-n}

    if [ "$response" != "y" ]; then
        print_info "跳过批量训练"
        return
    fi

    print_info "启动所有10个Fold的训练..."
    print_info "警告: 这可能需要很长时间"
    print_info "提示: 可以用 'nvidia-smi' 监控GPU使用"

    python3 kfold_shipsear_integration.py \
        --train-all \
        --gpus 4,5,6,7 \
        --splits-dir results/kfold_splits

    if [ $? -ne 0 ]; then
        print_error "部分或全部Fold训练失败"
    else
        print_success "所有Fold训练完成"
    fi
}

# 查看结果
show_results() {
    print_step "训练结果"

    if [ -f "results/kfold_cv_shipsear/kfold_shipsear_results.csv" ]; then
        print_info "显示结果:"
        cat results/kfold_cv_shipsear/kfold_shipsear_results.csv
    else
        print_info "尚未有训练结果"
    fi
}

# 主函数
main() {
    print_header "K-Fold十折叠交叉验证"

    # 显示选项
    echo ""
    echo "选择操作:"
    echo "  1) 完整流程（检查环境 -> 生成划分 -> 测试 -> 训练 -> 查看结果）"
    echo "  2) 只生成K-Fold划分"
    echo "  3) 测试单个Fold"
    echo "  4) 批量训练所有Fold"
    echo "  5) 查看训练结果"
    echo "  6) 帮助"
    echo ""

    read -p "请选择 (1-6, 默认1): " choice
    choice=${choice:-1}

    case $choice in
        1)
            check_dependencies
            check_files
            generate_kfold
            verify_splits
            test_single_fold
            run_all_folds
            show_results
            ;;
        2)
            check_dependencies
            check_files
            generate_kfold
            verify_splits
            ;;
        3)
            test_single_fold
            ;;
        4)
            run_all_folds
            ;;
        5)
            show_results
            ;;
        6)
            cat << 'HELP'
K-Fold十折叠交叉验证工具 - 帮助

完整文档:
  - KFOLD_SUMMARY.md         (总体汇总)
  - KFOLD_README.md          (快速入门)
  - KFOLD_USAGE_GUIDE.md     (详细参考)
  - KFOLD_TRAINING_GUIDE.md  (训练指南)

快速命令:
  1. 生成K-Fold划分:
     python3 kfold_cross_validation.py

  2. 测试单个Fold:
     python3 kfold_shipsear_integration.py --train-fold 0 --gpus 4,5,6,7

  3. 批量训练:
     python3 kfold_shipsear_integration.py --train-all --gpus 4,5,6,7

  4. 查看结果:
     python3 kfold_shipsear_integration.py --results

Python快速启动:
  python3 kfold_quick_start.py

查看详细错误:
  查看相关的日志文件或配置

更多帮助:
  查看文档中的"常见问题"部分
HELP
            ;;
        *)
            print_error "无效的选择"
            exit 1
            ;;
    esac

    print_header "完成"
    echo ""
    echo -e "${GREEN}✨ K-Fold十折叠交叉验证流程执行完成！${NC}"
    echo ""
}

# 执行
main
