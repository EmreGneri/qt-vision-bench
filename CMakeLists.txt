cmake_minimum_required(VERSION 3.21)
project(qt_vision_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt'nin moc/uic/rcc adimlarini CMake otomatik calistirsin
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

# Uyarilari acik tut. Sadece kendi hedeflerimize uygulanir; Qt/OpenCV basliklari
# -isystem ile geldigi icin onlarin uyarilari gurultu yapmiyor.
if (MSVC)
    set(PROJECT_WARNINGS /W4)
else()
    set(PROJECT_WARNINGS -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# vision_core: goruntu isleme hatti. Bilerek Qt'den bagimsiz tutuldu ki
# hem headless benchmark hem birim testler arayuz olmadan calisabilsin.
# ---------------------------------------------------------------------------
add_library(vision_core STATIC
    src/pipeline.cpp
    src/pipeline.h
)
target_include_directories(vision_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})
target_compile_options(vision_core PRIVATE ${PROJECT_WARNINGS})

# ---------------------------------------------------------------------------
# Uygulama. WIN32 bayragi bilerek verilmedi: --bench modu stdout'a JSON
# basiyor, konsol penceresi olmadan bu ciktiyi almak zorlasirdi.
# ---------------------------------------------------------------------------
add_executable(qt_vision_bench
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.h
    src/worker.cpp
    src/worker.h
)
target_link_libraries(qt_vision_bench PRIVATE vision_core Qt6::Widgets)
target_compile_options(qt_vision_bench PRIVATE ${PROJECT_WARNINGS})

# ---------------------------------------------------------------------------
# Testler
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE vision_core)

add_test(NAME pipeline COMMAND test_pipeline)
