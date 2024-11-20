#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>

//параметры разностной сетки
const double Lx = 1.0;//длина в пространстве по х
const double Lt = 1.0;//длина в пространстве по t 
const double dx = 0.01;//шаги по х
const double dt = 0.01;//шаги по t
const int Nx = static_cast<int>(Lx / dx) + 1;//кол узлов сетки
const int Nt = static_cast<int>(Lt / dt) + 1;

using namespace std;
// Начальное условие c(0, x) = 0.6 * cos(x)
double initial_condition(double x) {
    return 0.6 * cos(x);
}

// Функция правой части уравнения
double right_side(double t, double x) {
    return t + x;
}

int main() {
    setlocale(LC_ALL, "");
    //вектор значений
    vector<vector<double>> c(Nt, vector<double>(Nx, 0.0));
    double start_time, end_time;//для замера времени

    // Применение начального условия
    for (int i = 0; i < Nx; ++i) {
        c[0][i] = initial_condition(i * dx);
    }

    // Применение граничного условия
    for (int n = 0; n < Nt; ++n) {
        c[n][0] = 0.6;
    }
    //определяем количество потоков(2,4)
    omp_set_num_threads(2);
    // Расчет решения
    start_time = omp_get_wtime();
    //openMP для 2(4) потоков
#pragma omp parallel for
    for (int n = 0; n < Nt - 1; ++n) {
        for (int i = 1; i < Nx - 1; ++i) {
            c[n + 1][i] = c[n][i] + dt * (-4.0 * (c[n][i + 1] - c[n][i]) / dx + right_side(n * dt, i * dx));
        }
    }
    end_time = omp_get_wtime();

    // Вычисляем  время и ускорение
    double parallel_time = end_time - start_time;
    cout << "Время выполнения параллельной программы: " << parallel_time << " секунд." << endl;

    // Запуск последовательной версии без применения многопоточности для сравнения
    start_time = omp_get_wtime();
    for (int n = 0; n < Nt - 1; ++n) {
        for (int i = 1; i < Nx - 1; ++i) {
            c[n + 1][i] = c[n][i] + dt * (-4.0 * (c[n][i + 1] - c[n][i]) / dx + right_side(n * dt, i * dx));
        }
    }
    end_time = omp_get_wtime();
    double sequential_time = end_time - start_time;

    cout << "Время выполнения последовательной программы: " << sequential_time << " секунд." << endl;
    cout << "Ускорение: " << sequential_time / parallel_time << " раз." << endl;

    return 0;
}
