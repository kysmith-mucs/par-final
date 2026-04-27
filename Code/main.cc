/*
  Filename   : main.cc
  Author     : Kyle Smith, Andrew Elko, Evan Magill
  Course     : CMSC 476
  Date       : 2026-04-25
  Assignment : Final Project, Chladni Figure simulation with MPI.
  Description: Uses MPI to parallelize distributing "sand" grains, and randomly
               "bouncing" the grains with magnitudes scaled by the amplitude of
               vibrations at that point. After the specified number of
               iterations have occurred, local sections of the image are
               generated, then a global image is produced.
*/

/************************************************************/
// System includes

// Standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <print>
#include <random>
#include <ranges>
#include <vector>

// Image Magick
#include <Magick++.h>

// MPI
#include <mpi.h>

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

/************************************************************/
// Function prototypes/global vars/type definitions

struct Grain
{
  double x, y;
};

struct Input
{
  std::size_t grainCount;
  unsigned iterations;
  double m, n;
};

Input
retrieveInput (int ioSourceRank);

double
getAmplitude (double x, double y, double m, double n);

template<typename RandomNumberEngine>
std::vector<Grain>
populateLocalGrains (std::size_t localGrainCount, RandomNumberEngine& gen);

template<typename RandomNumberEngine>
void
performIterations (unsigned iterations,
                   std::vector<Grain>& localGrains,
                   RandomNumberEngine& gen,
                   double m,
                   double n);

int
getSectorRank (const Grain& grain, int commSize);

std::vector<std::vector<Grain>>
putGrainsInBuckets (const std::vector<Grain>& localGrains);

/************************************************************/

int
main ()
{
  MPI_Init (nullptr, nullptr);
  Magick::InitializeMagick (nullptr);

  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  const int ROOT_RANK { 0 };
  const int WIDTH { 1'024 };
  const int HEIGHT { 1'024 };
  Input in { retrieveInput (ROOT_RANK) };
  const std::size_t GRAINS_PER_PROC { in.grainCount /
                                      size }; // TODO proper partitioning.

  MPI_Barrier (MPI_COMM_WORLD);

  double lStart { MPI_Wtime () };

  std::random_device rd;
  std::mt19937 gen { rd () };

  std::vector<Grain> localGrains { populateLocalGrains (GRAINS_PER_PROC, gen) };

  performIterations (in.iterations, localGrains, gen, in.m, in.n);

  std::vector<std::vector<Grain>> buckets { putGrainsInBuckets (localGrains) };

  std::vector<int> send_counts (size), recv_counts (size);
  for (int i = 0; i < size; ++i)
    send_counts[i] = (int) buckets[i].size ();

  MPI_Alltoall (send_counts.data (),
                1,
                MPI_INT,
                recv_counts.data (),
                1,
                MPI_INT,
                MPI_COMM_WORLD);

  std::vector<int> s_displs (size, 0), r_displs (size, 0);
  for (int i = 1; i < size; ++i)
  {
    s_displs[i] = s_displs[i - 1] + send_counts[i - 1];
    r_displs[i] = r_displs[i - 1] + recv_counts[i - 1];
  }

  std::vector<Grain> send_buf;
  for (auto& b : buckets)
    send_buf.insert (send_buf.end (), b.begin (), b.end ());

  localGrains.resize (r_displs[size - 1] + recv_counts[size - 1]);

  int blocklengths[2] = { 1, 1 };
  MPI_Datatype structTypes[2] = { MPI_DOUBLE, MPI_DOUBLE };

  MPI_Aint offsets[2];
  offsets[0] = offsetof (Grain, x);
  offsets[1] = offsetof (Grain, y);

  MPI_Datatype MPI_GRAIN;

  MPI_Type_create_struct (2, blocklengths, offsets, structTypes, &MPI_GRAIN);
  MPI_Type_commit (&MPI_GRAIN);

  MPI_Alltoallv (send_buf.data (),
                 send_counts.data (),
                 s_displs.data (),
                 MPI_GRAIN,
                 localGrains.data (),
                 recv_counts.data (),
                 r_displs.data (),
                 MPI_GRAIN,
                 MPI_COMM_WORLD);

  MPI_Type_free (&MPI_GRAIN);

  std::vector<uint32_t> local_canvas (WIDTH * HEIGHT, 0);
  for (const auto& g : localGrains)
  {
    int px = std::clamp ((int) (g.x * WIDTH), 0, WIDTH - 1);
    int py = std::clamp ((int) (g.y * HEIGHT), 0, HEIGHT - 1);
    local_canvas[py * WIDTH + px]++;
  }

  std::vector<uint32_t> global_canvas;
  if (rank == ROOT_RANK)
    global_canvas.resize (WIDTH * HEIGHT);

  MPI_Reduce (local_canvas.data (),
              global_canvas.data (),
              WIDTH * HEIGHT,
              MPI_UINT32_T,
              MPI_SUM,
              0,
              MPI_COMM_WORLD);

  if (rank == ROOT_RANK)
  {
    Magick::Image image { Magick::Geometry (WIDTH, HEIGHT),
                          Magick::Color ("black") };

    uint32_t max_val = 0;
    for (auto v : global_canvas)
      if (v > max_val)
        max_val = v;

    for (int y = 0; y < HEIGHT; ++y)
    {
      for (int x = 0; x < WIDTH; ++x)
      {
        uint32_t count = global_canvas[y * WIDTH + x];
        if (count > 0)
        {
          double factor = std::log1p (count) / std::log1p (max_val);
          Magick::Quantum q = factor * 65'535.0;
          image.pixelColor (x, y, Magick::Color (q, q, q));
        }
      }
    }
    image.write ("chladni_result.png");
    std::println ("Pattern generated: chladni_result.png");
  }

  double lElapsed = MPI_Wtime () - lStart;
  double elapsed;
  MPI_Reduce (
    &lElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);
  if (rank == ROOT_RANK)
    std::println ("Time elapsed: {:.2f} Seconds", elapsed);

  MPI_Finalize ();
}

Input
retrieveInput (int ioSourceRank)
{
  int blocklengths[4] = { 1, 1, 1, 1 };
  MPI_Datatype structTypes[4] = {
    MPI_UNSIGNED_LONG, MPI_UNSIGNED, MPI_DOUBLE, MPI_DOUBLE
  };

  MPI_Aint offsets[4];
  offsets[0] = offsetof (Input, grainCount);
  offsets[1] = offsetof (Input, iterations);
  offsets[2] = offsetof (Input, m);
  offsets[3] = offsetof (Input, n);

  MPI_Datatype MPI_INPUT;

  MPI_Type_create_struct (4, blocklengths, offsets, structTypes, &MPI_INPUT);
  MPI_Type_commit (&MPI_INPUT);

  Input in;
  int myRank;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  if (myRank == ioSourceRank)
  {
    std::print ("Number of grains ==> ");
    std::cin >> in.grainCount;
    std::print ("Iterations ==> ");
    std::cin >> in.iterations;
    std::print ("m ==> ");
    std::cin >> in.m;
    std::print ("n ==> ");
    std::cin >> in.n;
  }
  MPI_Bcast (&in, 1, MPI_INPUT, ioSourceRank, MPI_COMM_WORLD);
  MPI_Type_free (&MPI_INPUT);
  return in;
}

double
getAmplitude (double x, double y, double m, double n)
{
  return std::abs (std::sin (n * M_PI * x) * std::sin (m * M_PI * y) -
                   std::sin (m * M_PI * x) * std::sin (n * M_PI * y));
}

template<typename RandomNumberEngine>
std::vector<Grain>
populateLocalGrains (std::size_t localGrainCount, RandomNumberEngine& gen)
{
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  std::uniform_real_distribution<double> dist_x (0.0, 1.0);
  double local_y_min { (double) rank / size };
  double local_y_max { (double) (rank + 1) / size };
  std::uniform_real_distribution<double> dist_y (local_y_min, local_y_max);

  std::vector<Grain> localGrains (localGrainCount);
  for (Grain& g : localGrains)
  {
    g.x = dist_x (gen);
    g.y = dist_y (gen);
  }
  return localGrains;
}

template<typename RandomNumberEngine>
void
performIterations (unsigned iterations,
                   std::vector<Grain>& localGrains,
                   RandomNumberEngine& gen,
                   double m,
                   double n)
{
  std::uniform_real_distribution<double> step_move (-0.01, 0.01);

  // Iterating over localGrains in the *outer* loop results in better locality.
  for (Grain& g : localGrains)
  {
    for ([[maybe_unused]] unsigned currentIteration :
         std::views::iota (0u, iterations))
    {
      double amplitude = getAmplitude (g.x, g.y, m, n);

      g.x = std::clamp (g.x + step_move (gen) * amplitude, 0.0, 1.0);
      g.y = std::clamp (g.y + step_move (gen) * amplitude, 0.0, 1.0);
    }
  }
}

int
getSectorRank (const Grain& grain, int commSize)
{
  return std::clamp ((int) (grain.y * commSize), 0, commSize - 1);
}

std::vector<std::vector<Grain>>
putGrainsInBuckets (const std::vector<Grain>& localGrains)
{
  int size;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  std::vector<std::vector<Grain>> buckets (size);
  for (Grain g : localGrains)
  {
    buckets[getSectorRank (g, size)].push_back (g);
  }
  return buckets;
}