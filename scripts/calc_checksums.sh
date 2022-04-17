#!/bin/bash

# CALC CPU CHECKSUMS
shasum $(ls output/cpu/*.v) >> output/cpu/cpu.vsum
shasum $(ls output/cpu/*.e) >> output/cpu/cpu.esum

# CALC PASCAL CHECKSUMS
shasum $(ls output/gpu/pascal/*.v) >> output/gpu/pascal/pascal.vsum
shasum $(ls output/gpu/pascal/*.e) >> output/gpu/pascal/pascal.esum

# CALC VOLTA CHECKSUMS
shasum $(ls output/gpu/volta/*.v) >> output/gpu/volta/volta.vsum
shasum $(ls output/gpu/volta/*.e) >> output/gpu/volta/volta.esum

# CALC AMPERE CHECKSUMS
shasum $(ls output/gpu/ampere/*.v) >> output/gpu/ampere/ampere.vsum
shasum $(ls output/gpu/ampere/*.e) >> output/gpu/ampere/ampere.esum

# PRINT CPU CHECKSUMS
cat $(ls output/cpu/*.vsum)
cat $(ls output/cpu/*.esum)

# PRINT GPU CHECKSUMS
cat $(ls output/gpu/*/*.vsum)
cat $(ls output/gpu/*/*.esum)