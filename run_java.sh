# File: run_java.sh
# Folder: root project directory
# --------------------------------------------------
# Java execution handler for compiled model in build/

BUILD_DIR="build"
CLASS_NAME="ModelEngine"

# Execute compiled Java model (flat structure, no package)
cd $BUILD_DIR || exit 1
java $CLASS_NAME
cd -
