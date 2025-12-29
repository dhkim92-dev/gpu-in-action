# helpers 디렉터리의 hpp 파일을 모두 포함
file(GLOB HELPERS_HEADERS
    ${CMAKE_SOURCE_DIR}/helpers/*.hpp
)

set(EXAMPLE_LISTS
    "1.vector-add",
    "2.reduce-sum",
    "3.dot-product",
    "4.vector-normalization",
    "5.prefix-sum-naive",
    "6.matrix-transpose",
    "7.matmul-naive",
)

foreach(EXAMPLE ${EXAMPLE_LISTS})

    # -------------------------
    # 실행 파일 이름용 예제명 가공
    # 1.vector-add -> vector-add
    # -------------------------
    string(REGEX REPLACE "^[0-9]+\\." "" EXAMPLE_NAME ${EXAMPLE})

    # 실행 파일 출력 디렉터리
    set(EXAMPLE_BUILD_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE}
    )
    file(MAKE_DIRECTORY ${EXAMPLE_BUILD_DIR})

    # =========================
    # OpenCL Example
    # =========================
    if(USE_OPENCL AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/OpenCL)

        file(GLOB OPENCL_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/OpenCL/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/OpenCL/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/OpenCL/*.hpp
        )

        if(OPENCL_SRC)
            set(OPENCL_EXE ${EXAMPLE_NAME}_OPENCL)
            add_executable(${OPENCL_EXE} ${OPENCL_SRC} ${HELPERS_HEADERS})
            target_link_libraries(${OPENCL_EXE} PRIVATE GPGPUBackend GPGPUHelpers)
            set_target_properties(${OPENCL_EXE} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BUILD_DIR}
            )

            # cl 파일 복사
            file(GLOB OPENCL_CL_FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/OpenCL/*.cl
            )
            if(OPENCL_CL_FILES)
                add_custom_target(copy_${OPENCL_EXE}_cl ALL
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${OPENCL_CL_FILES}
                            ${EXAMPLE_BUILD_DIR}
                )
                add_dependencies(${OPENCL_EXE} copy_${OPENCL_EXE}_cl)
            endif()
        endif()
    endif()

    # =========================
    # CUDA Example
    # =========================
    if(USE_CUDA AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/CUDA)
        
        file(GLOB CUDA_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/CUDA/*.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/CUDA/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/CUDA/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/CUDA/*.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/CUDA/*.cuh
        )

        if(CUDA_SRC)
            set(CUDA_EXE ${EXAMPLE_NAME}_CUDA)
            add_executable(${CUDA_EXE} ${CUDA_SRC} ${HELPERS_HEADERS})
            # 백엔드 + helper 전달
            target_link_libraries(${CUDA_EXE} PRIVATE GPGPUBackend GPGPUHelpers)
            set_target_properties(${CUDA_EXE} PROPERTIES
                CXX_STANDARD 17
                CXX_STANDARD_REQUIRED ON 
                CUDA_STANDARD 14
                CUDA_STANDARD_REQUIRED ON
                RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BUILD_DIR}
            )
        endif()
    endif()

endforeach()

