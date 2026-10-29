#[=======================================================================[.rst:
FindOnnxRuntime
---------------

Finds the ONNX Runtime library, either installed system-wide (e.g. the
Debian/Ubuntu package ``libonnxruntime-dev``) or extracted from an official
release archive, whose location can be passed via ``ONNXRUNTIME_ROOT`` (CMake
or environment variable).

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``OnnxRuntime::OnnxRuntime``
  The ONNX Runtime library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``OnnxRuntime_FOUND``
  True if the system has the ONNX Runtime library.
``OnnxRuntime_INCLUDE_DIRS``
  Include directories needed to use ONNX Runtime.
``OnnxRuntime_LIBRARIES``
  Libraries needed to link to ONNX Runtime.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OnnxRuntime_INCLUDE_DIR``
  The directory containing ``onnxruntime_cxx_api.h``.
``OnnxRuntime_LIBRARY``
  The path to the ONNX Runtime library.
``OnnxRuntime_DLL``
  The path to the ONNX Runtime DLL (Windows only).

#]=======================================================================]

if(NOT DEFINED ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_OnnxRuntime QUIET libonnxruntime)
endif()

find_path(
  OnnxRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include ${PC_OnnxRuntime_INCLUDE_DIRS}
  PATH_SUFFIXES onnxruntime onnxruntime/core/session
  DOC "ONNX Runtime include directory"
)
mark_as_advanced(OnnxRuntime_INCLUDE_DIR)

find_library(
  OnnxRuntime_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib ${PC_OnnxRuntime_LIBRARY_DIRS}
  DOC "ONNX Runtime library"
)
mark_as_advanced(OnnxRuntime_LIBRARY)

if(WIN32)
  find_file(
    OnnxRuntime_DLL
    NAMES onnxruntime.dll
    HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/bin
    DOC "ONNX Runtime DLL"
  )
  mark_as_advanced(OnnxRuntime_DLL)
endif()

if(DEFINED PC_OnnxRuntime_VERSION AND NOT PC_OnnxRuntime_VERSION STREQUAL "")
  set(OnnxRuntime_VERSION "${PC_OnnxRuntime_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR
  VERSION_VAR OnnxRuntime_VERSION
)

if(OnnxRuntime_FOUND)
  set(OnnxRuntime_LIBRARIES "${OnnxRuntime_LIBRARY}")
  set(OnnxRuntime_INCLUDE_DIRS "${OnnxRuntime_INCLUDE_DIR}")

  if(NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set_target_properties(
      OnnxRuntime::OnnxRuntime
      PROPERTIES
        IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}"
    )
  endif()
endif()
