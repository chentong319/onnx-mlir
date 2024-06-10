
#pragma once

namespace mlir {
class DialectRegistry;

void registerHISIMComputationOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace mlir

