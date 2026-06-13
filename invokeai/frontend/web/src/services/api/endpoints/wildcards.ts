import type { paths } from 'services/api/schema';
import type { S } from 'services/api/types';

import { api, buildV1Url, LIST_TAG } from '..';

export type WildcardRecordDTO = S['WildcardRecordDTO'];

/**
 * Builds an endpoint URL for the wildcards router
 * @example
 * buildWildcardsUrl('some-path')
 * // '/api/v1/wildcards/some-path'
 */
const buildWildcardsUrl = (path: string = '') => buildV1Url(`wildcards/${path}`);

const wildcardsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listWildcards: build.query<
      paths['/api/v1/wildcards/']['get']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildWildcardsUrl(),
      }),
      providesTags: ['FetchOnReconnect', { type: 'Wildcard', id: LIST_TAG }],
    }),
    createWildcard: build.mutation<
      paths['/api/v1/wildcards/']['post']['responses']['200']['content']['application/json'],
      S['WildcardWithoutId']
    >({
      query: (body) => ({
        url: buildWildcardsUrl(),
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'Wildcard', id: LIST_TAG }],
    }),
    updateWildcard: build.mutation<
      paths['/api/v1/wildcards/i/{wildcard_id}']['patch']['responses']['200']['content']['application/json'],
      { id: string; changes: S['WildcardChanges'] }
    >({
      query: ({ id, changes }) => ({
        url: buildWildcardsUrl(`i/${id}`),
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (response, error, { id }) => [
        { type: 'Wildcard', id: LIST_TAG },
        { type: 'Wildcard', id },
      ],
    }),
    deleteWildcard: build.mutation<void, string>({
      query: (wildcard_id) => ({
        url: buildWildcardsUrl(`i/${wildcard_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, wildcard_id) => [
        { type: 'Wildcard', id: LIST_TAG },
        { type: 'Wildcard', id: wildcard_id },
      ],
    }),
    importWildcards: build.mutation<
      paths['/api/v1/wildcards/import']['post']['responses']['200']['content']['application/json'],
      File[]
    >({
      query: (files) => {
        const formData = new FormData();
        for (const file of files) {
          formData.append('files', file);
        }

        return {
          url: buildWildcardsUrl('import'),
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: [{ type: 'Wildcard', id: LIST_TAG }],
    }),
  }),
});

export const {
  useListWildcardsQuery,
  useCreateWildcardMutation,
  useUpdateWildcardMutation,
  useDeleteWildcardMutation,
  useImportWildcardsMutation,
} = wildcardsApi;
