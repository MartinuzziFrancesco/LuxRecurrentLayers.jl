using LuxRecurrentLayers, Aqua, JET

Aqua.test_all(LuxRecurrentLayers; ambiguities=false, deps_compat=false)
JET.test_package(LuxRecurrentLayers; target_defined_modules=true)
