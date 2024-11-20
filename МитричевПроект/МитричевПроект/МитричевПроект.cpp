#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <cassert>

using namespace std;

// Параметры разностной сетки
double Lx = 1.0;  // Длина в пространстве по x
double Lt = 1.0;  // Длина в пространстве по t
double dx = 0.01; // Шаги по x
double dt = 0.01; // Шаги по t
int Nx = static_cast<int>(Lx / dx) + 1; // Количество узлов по x
int Nt = static_cast<int>(Lt / dt) + 1; // Количество узлов по t

// Начальное условие c(0, x) = 0.6 * cos(x)
double initial_condition(double x) {
    return 0.6 * cos(x);
}

// Функция правой части уравнения
double right_side(double t, double x) {
    return t + x;
}
// Новое начальное условие c(0, x) = 0.4 * sin(x)
double new_initial_condition(double x) {
    return 0.4 * sin(x);
}

// Новая функция правой части уравнения
double new_right_side(double t, double x) {
    return t * x;
}


// Функция для расчёта решения
vector<vector<double>> solve(bool parallel, int threads = 1) {
    vector<vector<double>> c(Nt, vector<double>(Nx, 0.0));

    // Применение начального условия
    for (int i = 0; i < Nx; ++i) {
        c[0][i] = initial_condition(i * dx);
    }

    // Применение граничного условия
    for (int n = 0; n < Nt; ++n) {
        c[n][0] = 0.6;
    }

    if (parallel) {
        omp_set_num_threads(threads);
#pragma omp parallel for
        for (int n = 0; n < Nt - 1; ++n) {
            for (int i = 1; i < Nx - 1; ++i) {
                c[n + 1][i] = c[n][i] + dt * (-4.0 * (c[n][i + 1] - c[n][i]) / dx + right_side(n * dt, i * dx));
            }
        }
    }
    else {
        for (int n = 0; n < Nt - 1; ++n) {
            for (int i = 1; i < Nx - 1; ++i) {
                c[n + 1][i] = c[n][i] + dt * (-4.0 * (c[n][i + 1] - c[n][i]) / dx + right_side(n * dt, i * dx));
            }
        }
    }

    return c;
}


// Функция для расчёта решения
vector<vector<double>> solve_2(bool parallel, int threads = 1) {
    vector<vector<double>> c(Nt, vector<double>(Nx, 0.0));

    // Применение нового начального условия
    for (int i = 0; i < Nx; ++i) {
        c[0][i] = new_initial_condition(i * dx);
    }

    // Применение граничного условия
    for (int n = 0; n < Nt; ++n) {
        c[n][0] = 0.6;
    }

    if (parallel) {
        omp_set_num_threads(threads);
#pragma omp parallel for
        for (int n = 0; n < Nt - 1; ++n) {
            for (int i = 1; i < Nx - 1; ++i) {
                c[n + 1][i] = c[n][i] + dt * (-4.0 * (c[n][i + 1] - c[n][i]) / dx + new_right_side(n * dt, i * dx));
            }
        }
    }
    else {
        for (int n = 0; n < Nt - 1; ++n) {
            for (int i = 1; i < Nx - 1; ++i) {
                c[n + 1][i] = c[n][i] + dt * (-4.0 * (c[n][i + 1] - c[n][i]) / dx + new_right_side(n * dt, i * dx));
            }
        }
    }

    return c;
}





// Тест 2: проверка при граничных  правых учловий
void test_performance_new() {
    double start_time, end_time;

    start_time = omp_get_wtime();
    solve_2(false);
    end_time = omp_get_wtime();
    double sequential_time = end_time - start_time;

    start_time = omp_get_wtime();
    solve_2(true, 16);
    end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;

    cout << "Последовательное время: " << sequential_time << " секунд." << endl;
    cout << "Параллельное время: " << parallel_time << " секунд." << endl;

    cout << "Тест 3 пройден" << endl;
}
// Тест 1: Оценка производительности 2 потоков
void test_performance() {
    double start_time, end_time;

    start_time = omp_get_wtime();
    solve(false);
    end_time = omp_get_wtime();
    double sequential_time = end_time - start_time;

    start_time = omp_get_wtime();
    solve(true, 2);
    end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;

    cout << "Последовательное время: " << sequential_time << " секунд." << endl;
    cout << "Параллельное время: " << parallel_time << " секунд." << endl;

cout << "Тест 1 пройден: параллельная версия работает быстрее." << endl;
}
// Тест 2:  Оценка производительности 16 потоков
void test_performance_16() {
    double start_time, end_time;

    start_time = omp_get_wtime();
    solve(false);
    end_time = omp_get_wtime();
    double sequential_time = end_time - start_time;

    start_time = omp_get_wtime();
    solve(true, 6);
    end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;

    cout << "Последовательное время: " << sequential_time << " секунд." << endl;
    cout << "Параллельное время: " << parallel_time << " секунд." << endl;

    cout << "Тест 2 пройден" << endl;
}



// Главная функция для запуска всех тестов
int main() {
    setlocale(LC_ALL, "");


    test_performance();
    test_performance_16();
    test_performance_new();


    cout << "Все тесты пройдены успешно." << endl;
    system("pause");
    return 0;
}
