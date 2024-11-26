#include <libmeshb7.h>
#include <stdio.h>
#include <stdlib.h>

int64_t open_file_write(const char *fname, int dim)
{

    const int version = 3;
    return GmfOpenMesh(fname, GmfWrite, version, dim);
}

int64_t open_file_read(const char *fname, int *version, int *dim)
{

    return GmfOpenMesh(fname, GmfRead, version, dim);
}

void close_file(int64_t file)
{

    GmfCloseMesh(file);
}

int write_vertices(int64_t file, int64_t n_verts, int dim, const double *coords)
{
    GmfSetKwd(file, GmfVertices, n_verts);
    for (int i = 0; i < n_verts; i++)
    {
        if (dim == 3)
        {
            GmfSetLin(file, GmfVertices, coords[3 * i],
                      coords[3 * i + 1], coords[3 * i + 2], i + 1);
        }
        else if (dim == 2)
        {
            GmfSetLin(file, GmfVertices, coords[2 * i],
                      coords[2 * i + 1], i + 1);
        }
        else
        {
            return -dim;
        }
    }
    return 0;
}

int write_elements(int64_t file, int64_t n_elems, int m, int etype, const int64_t *conn, const int *tags)
{
    GmfSetKwd(file, etype, n_elems);
    for (int i = 0; i < n_elems; i++)
    {
        if (m == 2)
        {
            GmfSetLin(file, etype, conn[2 * i] + 1,
                      conn[2 * i + 1] + 1, tags[i]);
        }
        else if (m == 3)
        {
            GmfSetLin(file, etype, conn[3 * i] + 1,
                      conn[3 * i + 1] + 1, conn[3 * i + 2] + 1, tags[i]);
        }
        else if (m == 6)
        {
            GmfSetLin(file, etype, conn[6 * i] + 1,
                      conn[6 * i + 1] + 1, conn[6 * i + 2] + 1, conn[6 * i + 3] + 1,
                      conn[6 * i + 4] + 1, conn[6 * i + 5] + 1, tags[i]);
        }
        else
        {
            return -m;
        }
    }
    return 0;
}

int write_sol(int64_t file, int dim, int loc, int64_t n, int m, const double *sol)
{
    int sol_type;
    if (m == 1)
    {
        sol_type = GmfSca;
    }
    else if (m == dim)
    {
        sol_type = GmfVec;
    }
    else if (m == dim * (dim + 1) / 2)
    {
        sol_type = GmfSymMat;
    }
    else if (m == dim * dim)
    {
        sol_type = GmfMat;
    }
    else
    {
        return -1;
    }
    GmfSetKwd(file, loc, n, 1, &sol_type);

    for (int i = 0; i < n; i++)
    {
        GmfSetLin(file, loc, sol + m * i);
    }
    return 0;
}

int64_t get_num_entities(int64_t file, int kwd)
{
    return GmfStatKwd(file, kwd);
}

int64_t get_sol_info(int64_t file, int loc, int *m)
{
    int n_sols;
    int sol_types[GmfMaxTyp];
    return GmfStatKwd(file, loc, &n_sols, m, sol_types);
}

int read_vertices(int64_t file, int64_t n_verts, int dim, double *coords)
{
    int tag;
    GmfGotoKwd(file, GmfVertices);
    for (int i = 0; i < n_verts; i++)
    {
        if (dim == 3)
        {
            GmfGetLin(file, GmfVertices, &coords[3 * i],
                      &coords[3 * i + 1], &coords[3 * i + 2], &tag);
        }
        else if (dim == 2)
        {
            GmfGetLin(file, GmfVertices, &coords[2 * i],
                      &coords[2 * i + 1], &tag);
        }
        else
        {
            return -dim;
        }
    }
    return 0;
}

int read_elements(int64_t file, int64_t n_elems, int m, int etype, int64_t *conn, int *tags)
{
    GmfGotoKwd(file, etype);
    for (int i = 0; i < n_elems; i++)
    {
        if (m == 2)
        {
            GmfGetLin(file, etype, &conn[2 * i],
                      &conn[2 * i + 1], &tags[i]);
        }
        else if (m == 3)
        {
            GmfGetLin(file, etype, &conn[3 * i],
                      &conn[3 * i + 1], &conn[3 * i + 2], &tags[i]);
        }
        else if (m == 6)
        {
            GmfGetLin(file, etype, &conn[6 * i],
                      &conn[6 * i + 1], &conn[6 * i + 2], &conn[6 * i + 3],
                      &conn[6 * i + 4], &conn[6 * i + 5], &tags[i]);
        }
        else
        {
            return -m;
        }
    }
    for (int i = 0; i < m * n_elems; i++)
    {
        conn[i] -= 1;
    }
    return 0;
}

int read_sol(int64_t file, int loc, int64_t n, int m, const double *sol)
{
    GmfGotoKwd(file, loc);

    for (int i = 0; i < n; i++)
    {
        GmfGetLin(file, loc, sol + m * i);
    }
    return 0;
}
