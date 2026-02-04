#!/bin/bash

# 1. 变量定义
PROJECT_PATH="/mnt/d/thu-project/TinyInfiniTrain-master/TinyInfiniTrain-master"
PASSWORD="Gjdyyaez2816"
EMAIL="you@example.com"
USERNAME="YourName"
GIT_REMOTE="git@github.com:SkyHigh-achieving/TinyInfiniTrain.git"

echo "=== 1. 进入项目目录 ==="
cd "$PROJECT_PATH" || exit 1
echo "当前目录: $(pwd)"

echo "=== 2. 修复 cmake 安装 ==="
# 使用 echo 传递密码给 sudo -S
echo "$PASSWORD" | sudo -S apt update
echo "$PASSWORD" | sudo -S apt install -y build-essential cmake ninja-build pkg-config

echo "验证版本:"
cmake --version

echo "=== 3. 构建流程 ==="
rm -rf build
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja

# 验证生成的可执行文件 (根据项目结构，可能是 TinyInfiniTrain 或测试程序)
if [ -f "./TinyInfiniTrain" ]; then
    echo "构建成功: TinyInfiniTrain 已生成"
else
    echo "警告: 未找到 TinyInfiniTrain 可执行文件，检查 build 目录内容:"
    ls -F
fi
cd ..

echo "=== 4. GitHub 配置 ==="
git config --global user.name "$USERNAME"
git config --global user.email "$EMAIL"

if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -C "$EMAIL" -f ~/.ssh/id_ed25519 -q -N ""
fi

echo "--- SSH 公钥 (请将其添加到 GitHub) ---"
cat ~/.ssh/id_ed25519.pub
echo "--------------------------------------"

# 测试连通性 (跳过交互式确认)
ssh -o StrictHostKeyChecking=no -T git@github.com 2>&1 | grep "successfully authenticated" || echo "SSH 测试提示: 请确保已将上述公钥添加到 GitHub"

echo "=== 5. 推送代码 ==="
# 如果尚未初始化
if [ ! -d ".git" ]; then
    git init
fi

if git remote | grep -q "origin"; then
    git remote remove origin
fi
git remote add origin "$GIT_REMOTE"
git add .
git commit -m "init: complete cmake build & WSL setup"
git branch -M main

echo "尝试推送 (如果 SSH Key 已添加则会成功)..."
git push -u origin main || echo "推送失败: 请先在 GitHub 添加 SSH Key"
