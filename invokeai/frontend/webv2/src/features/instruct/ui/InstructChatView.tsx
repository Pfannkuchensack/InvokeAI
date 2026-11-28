/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop -- one chat per project; tiles and steps are memoized, so only what changed renders. */
import type { GalleryItem } from '@features/gallery';
import type { ImageWithDims } from '@features/generation/contracts';
import type { InstructNode, InstructTreeView } from '@features/instruct/core/types';
import type { KeyboardEvent, ReactNode } from 'react';

import { Box, Flex, HStack, Icon, Image, Progress, SimpleGrid, Spinner, Stack, Text, Textarea } from '@chakra-ui/react';
import { GalleryMediaSlot } from '@features/gallery/mediaSlot';
import { galleryImageUrls } from '@features/gallery/utility';
import {
  subscribeInstructEntry,
  takeInstructEntry,
  type InstructEntryRequest,
} from '@features/instruct/core/entryRequests';
import {
  getInstructRoute,
  getInstructSteps,
  getInstructTip,
  getNearestCompletedAncestor,
  type InstructStepView,
} from '@features/instruct/core/history';
import { getReferenceNumber } from '@features/instruct/core/references';
import { getInstructValidationReasons } from '@features/instruct/core/values';
import { useInstructActions, useInstructUi } from '@features/instruct/ui/InstructUiContext';
import { useRegisterDraftFlusher } from '@platform/react/draftRegistry';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { isModalPresent } from '@platform/ui/modalPresence';
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CornerDownRightIcon,
  ExpandIcon,
  ImageOffIcon,
  RotateCcwIcon,
  SendHorizontalIcon,
  WandSparklesIcon,
  XIcon,
} from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useEnsureInstructCatalog, useInstructValues } from './useInstructValues';

const ACCEPT_IMAGES = ['image'] as const;
const COMMIT_DELAY_MS = 400;
/** Keeps several steps on screen whatever size the results are. */
const TILE_MAX_HEIGHT = '340px';
const CHAT_MAX_WIDTH = '760px';
const TILE_ACTIONS = { opacity: 1, pointerEvents: 'auto' } as const;
const TILE_HOVER_ACTIONS = { '&:hover [data-tile-actions], &:focus-within [data-tile-actions]': TILE_ACTIONS };

const toImage = (item: GalleryItem | null): ImageWithDims | null =>
  item?.kind === 'image' ? { height: item.height, image_name: item.name, width: item.width } : null;

/** Scrolls to the end of the history when it mounts; keyed by the last step, so only a new step brings it back. */
const scrollIntoViewOnMount = (element: HTMLElement | null) => element?.scrollIntoView?.({ block: 'end' });

/** Scrolls the tile of a node into view when it mounts; keyed per jump. */
const ScrollToNode = ({ nodeId }: { nodeId: string }) => {
  const scroll = useCallback(
    (element: HTMLElement | null) =>
      element
        ?.closest('[data-instruct-history]')
        ?.querySelector(`[data-node-id="${CSS.escape(nodeId)}"]`)
        ?.scrollIntoView?.({ block: 'center' }),
    [nodeId]
  );

  return <Box ref={scroll} />;
};

/** Starts or replaces the history from an image; a replacement waits for the user's confirmation. */
/** Cancels a step: its run here, or, for a step this browser holds no run for, waiting for it after a confirmation. */
const useCancelStep = () => {
  const actions = useInstructActions();
  const { t } = useTranslation();
  const [orphan, setOrphan] = useState<string | null>(null);
  const cancel = useCallback(
    (stepId: string, run: 'queued' | 'running' | undefined) => (run ? actions.cancelStep(stepId) : setOrphan(stepId)),
    [actions]
  );
  const dialog = (
    <ConfirmDialog
      body={t('widgets.instruct.abandonBody')}
      confirmLabel={t('widgets.instruct.abandonConfirm')}
      isDestructive={false}
      isOpen={orphan !== null}
      title={t('widgets.instruct.abandonTitle')}
      onClose={() => setOrphan(null)}
      onConfirm={() => {
        if (orphan) {
          actions.abandonStep(orphan);
        }
        setOrphan(null);
      }}
    />
  );

  return { cancel, dialog };
};

/** What an entry carries beyond the image: see `InstructEntryRequest`. */
type StartOptions = Pick<InstructEntryRequest, 'continueFromImage' | 'values' | 'valuesIfKept'>;

const useStartFromImage = (onJump?: (nodeId: string) => void) => {
  const ui = useInstructUi();
  const actions = useInstructActions();
  const { t } = useTranslation();
  const [pending, setPending] = useState<{ image: ImageWithDims; options: StartOptions } | null>(null);
  const jumpTo = useCallback(
    (tree: InstructTreeView, nodeId: string, options: StartOptions) => {
      const route = getInstructRoute(tree, nodeId);

      route.slice(0, -1).forEach((node, index) => actions.selectChild(node.id, route[index + 1]!.id));
      // A node with steps below it is where the next step branches off; the end of a branch needs no mark, unless a
      // recall asks for it.
      actions.patchValues({
        continueFromNodeId:
          options.continueFromImage || tree.nodes.some((node) => node.parentId === nodeId) ? nodeId : null,
      });
      onJump?.(nodeId);
    },
    [actions, onJump]
  );
  const request = useCallback(
    (image: ImageWithDims | null, options: StartOptions = {}) => {
      if (!image) {
        return;
      }

      const existing = ui.tree?.nodes.find((node) => node.imageName === image.image_name);

      // A newer request replaces one still waiting on the replace dialog, values included.
      setPending(null);

      if (!ui.tree) {
        actions.startFromImage(image);
        if (options.values) {
          actions.patchValues(options.values);
        }
      } else if (existing) {
        jumpTo(ui.tree, existing.id, options);
        if (options.values) {
          actions.patchValues(options.values);
        }
      } else {
        setPending({ image, options });
      }
    },
    [actions, jumpTo, ui.tree]
  );
  const { projectId } = ui;
  const takeEntry = useCallback(() => {
    const entry = takeInstructEntry(projectId);

    if (entry) {
      request(entry.image, {
        continueFromImage: entry.continueFromImage,
        values: entry.values,
        valuesIfKept: entry.valuesIfKept,
      });
    }
  }, [projectId, request]);

  // An image sent here from elsewhere (gallery, canvas, recall) waits for this view; one sent while it shows is taken now.
  useMountEffect(takeEntry);
  useEffect(() => subscribeInstructEntry(takeEntry), [takeEntry]);

  // The dialog closes after a confirmation too; only a dismissal still has the recall's values to apply.
  const isConfirmedRef = useRef(false);
  const applyValues = (values: StartOptions['values']) => {
    if (values) {
      actions.patchValues(values);
    }
  };
  const confirm = () => {
    if (pending) {
      isConfirmedRef.current = true;
      actions.startFromImage(pending.image);
      applyValues(pending.options.values);
    }
  };
  const dismiss = () => {
    // A recall's values apply even when the user keeps the old history.
    if (pending && !isConfirmedRef.current) {
      applyValues(pending.options.valuesIfKept ?? pending.options.values);
    }
    isConfirmedRef.current = false;
    setPending(null);
  };
  const dialog = (
    <ConfirmDialog
      body={t('widgets.instruct.replaceBody')}
      confirmLabel={t('widgets.instruct.replaceConfirm')}
      isDestructive={false}
      isOpen={pending !== null}
      title={t('widgets.instruct.replaceTitle')}
      onClose={dismiss}
      onConfirm={confirm}
    />
  );

  return { dialog, request };
};

/** Whether a paste lands in a text field, which keeps its own paste. */
const isTextTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement && (target.isContentEditable || target.matches('input, textarea, select'));

const EmptyView = () => {
  const { t } = useTranslation();
  const { hasUnreadableHistory, pasteShortcut } = useInstructUi();
  const actions = useInstructActions();
  const [isUploading, setIsUploading] = useState(false);

  // While the empty Edit view is the one shown (a hidden view unmounts its effects), a pasted image starts the history
  // wherever focus is. The capture phase takes it before the workbench's own paste dialog, which listens on the document.
  useMountEffect(() => {
    const paste = (event: globalThis.ClipboardEvent) => {
      const file = Array.from(event.clipboardData?.files ?? []).find((candidate) =>
        candidate.type.startsWith('image/')
      );

      // Nothing is uploaded for a history that cannot be started over anyway.
      if (!file || hasUnreadableHistory || isModalPresent() || isTextTarget(event.target) || event.defaultPrevented) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      setIsUploading(true);
      actions
        .uploadFile(file)
        .then((image) => (image ? actions.startFromImage(image) : undefined))
        .catch(() => actions.reportError(t('widgets.instruct.uploadFailed')))
        .finally(() => setIsUploading(false));
    };

    document.addEventListener('paste', paste, true);

    return () => document.removeEventListener('paste', paste, true);
  });

  const { dialog, request } = useStartFromImage();

  return (
    <Flex align="center" flex="1" justify="center" minH="0" p="6">
      <Stack align="center" gap="4" maxW="360px" textAlign="center" w="full">
        <Icon as={WandSparklesIcon} boxSize="8" color="fg.muted" />
        <Stack gap="1">
          <Text fontSize="lg" fontWeight="600">
            {t('widgets.instruct.emptyTitle')}
          </Text>
          <Text color="fg.muted" fontSize="sm" textWrap="pretty">
            {t('widgets.instruct.emptyBody')}
          </Text>
        </Stack>
        {hasUnreadableHistory ? (
          <Text color="fg.warning" fontSize="sm" role="status" textWrap="pretty">
            {t('widgets.instruct.historyUnreadable')}
          </Text>
        ) : (
          <>
            <Box w="full">
              <GalleryMediaSlot
                accept={ACCEPT_IMAGES}
                busy={isUploading}
                dropId="instruct-start-slot"
                labels={{ choose: t('widgets.instruct.chooseImage'), drop: t('widgets.instruct.dropImage') }}
                layout="tile"
                value={null}
                onChange={(item) => request(toImage(item))}
              />
            </Box>
            <Text color="fg.muted" fontSize="xs">
              {t('widgets.instruct.pasteHint', { keys: pasteShortcut })}
            </Text>
          </>
        )}
      </Stack>
      {dialog}
    </Flex>
  );
};

const TileAction = ({ icon, label, onClick }: { icon: typeof XIcon; label: string; onClick: () => void }) => (
  <IconButton aria-label={label} bg="bg/80" size="xs" tooltip={label} variant="subtle" onClick={onClick}>
    <Icon as={icon} />
  </IconButton>
);

const PlaceholderTile = ({ children, height, width }: { children: ReactNode; height: number; width: number }) => (
  <Flex
    align="center"
    aspectRatio={`${width} / ${height}`}
    bg="bg.muted"
    maxH={TILE_MAX_HEIGHT}
    borderColor="border.subtle"
    borderWidth="1px"
    direction="column"
    gap="1.5"
    justify="center"
    p="3"
    rounded="md"
    textAlign="center"
    w="full"
  >
    {children}
  </Flex>
);

/** `choice` is set for a variant among several siblings; the source and a lone result are not choices. */
const VariantTile = memo(function VariantTile({
  choice,
  isContinueTarget,
  node,
  parentId,
  run,
}: {
  choice?: { isSelected: boolean; number: number; total: number };
  isContinueTarget: boolean;
  node: InstructNode;
  parentId: string;
  run: 'queued' | 'running' | undefined;
}) {
  const { t } = useTranslation();
  const actions = useInstructActions();
  const { missingImages } = useInstructUi();
  const select = useCallback(() => actions.selectChild(parentId, node.id), [actions, node.id, parentId]);

  if (node.status === 'pending') {
    return (
      <PlaceholderTile height={node.height} width={node.width}>
        <Spinner size="md" />
        <Text color="fg.muted" fontSize="sm">
          {run === 'running' ? t('widgets.instruct.generating') : t('widgets.instruct.queued')}
        </Text>
        <Progress.Root maxW="24" size="md" value={null} w="full">
          <Progress.Track>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
      </PlaceholderTile>
    );
  }

  if (node.status === 'failed') {
    return (
      <PlaceholderTile height={node.height} width={node.width}>
        <Icon as={ImageOffIcon} boxSize="5" color="fg.error" />
        <Text color="fg.error" fontSize="sm" fontWeight="500">
          {t('widgets.instruct.failed')}
        </Text>
        {node.error ? (
          <Text color="fg.muted" fontSize="xs" lineClamp={4} role="alert" textWrap="pretty">
            {node.error}
          </Text>
        ) : null}
      </PlaceholderTile>
    );
  }

  if (node.status === 'canceled' || node.status === 'deleted' || !node.imageName) {
    return (
      <PlaceholderTile height={node.height} width={node.width}>
        <Icon as={node.status === 'canceled' ? XIcon : ImageOffIcon} boxSize="5" color="fg.muted" />
        <Text color="fg.muted" fontSize="sm">
          {node.status === 'canceled' ? t('widgets.instruct.canceled') : t('widgets.instruct.deleted')}
        </Text>
      </PlaceholderTile>
    );
  }

  const imageName = node.imageName;

  // Not a deleted image (that is stored): this session could not find it, perhaps because it is not shown to this user.
  if (missingImages.has(imageName)) {
    const unavailable = (
      <PlaceholderTile height={node.height} width={node.width}>
        <Icon as={ImageOffIcon} boxSize="5" color="fg.muted" />
        <Text color="fg.muted" fontSize="sm">
          {t('widgets.instruct.unavailable')}
        </Text>
      </PlaceholderTile>
    );

    return choice ? (
      <Box
        aria-label={t('widgets.instruct.variantOf', { number: choice.number, total: choice.total })}
        aria-pressed={choice.isSelected}
        as="button"
        cursor="default"
        data-node-id={node.id}
        onClick={select}
      >
        {unavailable}
      </Box>
    ) : (
      <Box data-node-id={node.id}>{unavailable}</Box>
    );
  }

  const picture = (
    <Image alt="" h="full" loading="lazy" objectFit="contain" src={galleryImageUrls.thumbnail(imageName)} w="full" />
  );
  const frame = {
    aspectRatio: `${node.width} / ${node.height}`,
    bg: 'bg.muted',
    display: 'block',
    maxH: TILE_MAX_HEIGHT,
    w: 'full',
  } as const;

  return (
    <Box
      aria-current={choice?.isSelected ? 'true' : undefined}
      borderColor={choice?.isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth={choice?.isSelected ? '2px' : '1px'}
      css={TILE_HOVER_ACTIONS}
      data-node-id={node.id}
      overflow="hidden"
      position="relative"
      rounded="md"
    >
      {choice ? (
        <Box
          {...frame}
          aria-label={t('widgets.instruct.variantOf', { number: choice.number, total: choice.total })}
          aria-pressed={choice.isSelected}
          as="button"
          cursor="default"
          onClick={select}
        >
          {picture}
        </Box>
      ) : (
        <Box {...frame}>{picture}</Box>
      )}
      {isContinueTarget ? (
        <Text
          bg="accent.solid"
          color="accent.contrast"
          fontSize="xs"
          left="1.5"
          position="absolute"
          px="1.5"
          rounded="sm"
          top="1.5"
        >
          {t('widgets.instruct.continuingHere')}
        </Text>
      ) : null}
      <HStack
        data-tile-actions
        gap="1"
        opacity={0}
        pointerEvents="none"
        position="absolute"
        right="1.5"
        top="1.5"
        transition="opacity var(--wb-motion-duration-fast) ease"
        _focusWithin={TILE_ACTIONS}
      >
        <TileAction icon={ExpandIcon} label={t('widgets.instruct.view')} onClick={() => actions.openImage(imageName)} />
        <TileAction
          icon={CornerDownRightIcon}
          label={t('widgets.instruct.continueHere')}
          onClick={() => {
            if (parentId !== node.id) {
              select();
            }
            actions.patchValues({ continueFromNodeId: node.id });
          }}
        />
      </HStack>
    </Box>
  );
});

const StepSwitcher = ({ step }: { step: InstructStepView }) => {
  const { t } = useTranslation();
  const actions = useInstructActions();
  const position = step.siblingSteps.findIndex((entry) => entry.stepId === step.stepId);
  const move = (offset: number) => {
    const target = step.siblingSteps[position + offset];

    if (target) {
      actions.selectChild(step.parent.id, target.firstNodeId);
    }
  };

  return (
    <HStack gap="1" role="group" aria-label={t('widgets.instruct.versions')}>
      <IconButton
        aria-label={t('widgets.instruct.previousStep')}
        disabled={position <= 0}
        size="sm"
        tooltip={t('widgets.instruct.previousStep')}
        variant="outline"
        onClick={() => move(-1)}
      >
        <ChevronLeftIcon />
      </IconButton>
      <Text color="fg.muted" fontSize="sm" fontVariantNumeric="tabular-nums" minW="10" textAlign="center">
        {position + 1} / {step.siblingSteps.length}
      </Text>
      <IconButton
        aria-label={t('widgets.instruct.nextStep')}
        disabled={position >= step.siblingSteps.length - 1}
        size="sm"
        tooltip={t('widgets.instruct.nextStep')}
        variant="outline"
        onClick={() => move(1)}
      >
        <ChevronRightIcon />
      </IconButton>
    </HStack>
  );
};

const StepMessage = memo(function StepMessage({
  continueFromNodeId,
  run,
  step,
}: {
  continueFromNodeId: string | null;
  run: 'queued' | 'running' | undefined;
  step: InstructStepView;
}) {
  const { t } = useTranslation();
  const actions = useInstructActions();
  const { cancel, dialog: cancelDialog } = useCancelStep();
  const { missingImages } = useInstructUi();
  const first = step.variants[0]!;
  const isRunning = step.variants.some((node) => node.status === 'pending');
  const canRerun =
    !isRunning &&
    step.variants.every((node) => node.status === 'failed' || node.status === 'canceled' || node.status === 'deleted');

  return (
    <Stack gap="2">
      {cancelDialog}
      <HStack align="center" gap="3" justify="flex-end">
        {step.siblingSteps.length > 1 ? <StepSwitcher step={step} /> : null}
        <Stack bg="bg.muted" gap="1.5" maxW="85%" p="2.5" rounded="lg">
          <Text fontSize="sm" textWrap="pretty" whiteSpace="pre-wrap">
            {first.instruction}
          </Text>
          {first.referenceImages.length > 0 ? (
            <HStack flexWrap="wrap" gap="1.5">
              {first.referenceImages.map((reference, index) => (
                <HStack key={reference.imageName} bg="bg.subtle" gap="1" pe="1.5" rounded="sm">
                  {missingImages.has(reference.imageName) ? (
                    <Icon aria-label={t('widgets.instruct.referenceMissing')} as={ImageOffIcon} boxSize="5" p="0.5" />
                  ) : (
                    <Image
                      alt=""
                      boxSize="5"
                      objectFit="cover"
                      roundedStart="sm"
                      src={galleryImageUrls.thumbnail(reference.imageName)}
                    />
                  )}
                  <Text fontSize="xs">{t('widgets.instruct.imageN', { number: getReferenceNumber(index) })}</Text>
                </HStack>
              ))}
            </HStack>
          ) : null}
        </Stack>
      </HStack>
      <Stack gap="1.5">
        <HStack justify="flex-end" minH="6">
          <HStack gap="1">
            {isRunning ? (
              <Button size="xs" variant="outline" onClick={() => cancel(step.stepId, run)}>
                {t('common.cancel')}
              </Button>
            ) : null}
            {canRerun ? (
              <Button size="xs" variant="outline" onClick={() => void actions.runAgain(first.id)}>
                <RotateCcwIcon />
                {t('widgets.instruct.runAgain')}
              </Button>
            ) : null}
          </HStack>
        </HStack>
        <SimpleGrid
          columns={step.variants.length > 1 ? 2 : 1}
          gap="2"
          maxW={step.variants.length > 1 ? '700px' : '360px'}
        >
          {step.variants.map((node, index) => (
            <VariantTile
              key={node.id}
              choice={
                step.variants.length > 1
                  ? { isSelected: node.id === step.selected.id, number: index + 1, total: step.variants.length }
                  : undefined
              }
              isContinueTarget={continueFromNodeId === node.id}
              node={node}
              parentId={step.parent.id}
              run={run}
            />
          ))}
        </SimpleGrid>
      </Stack>
    </Stack>
  );
});

const Composer = ({ tree }: { tree: InstructTreeView }) => {
  const { t } = useTranslation();
  const ui = useInstructUi();
  const actions = useInstructActions();
  const values = useInstructValues();
  const { cancel, dialog: cancelDialog } = useCancelStep();
  const tip = getInstructTip(tree, values);
  // What is typed shows until it is committed; an instruction set elsewhere shows as it is stored.
  const [draft, setDraft] = useState<string | null>(null);
  const draftRef = useRef<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isLocked = tip.status === 'pending';
  // A step cannot continue from an image this session could not find, any more than from a deleted one.
  const isTipUnavailable =
    tip.status !== 'completed' || (tip.imageName !== null && ui.missingImages.has(tip.imageName));
  const text = draft ?? values.instruction;

  const commit = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }

    const pending = draftRef.current;

    if (pending !== null) {
      draftRef.current = null;
      setDraft(null);
      actions.patchValues({ instruction: pending });
    }
  }, [actions, setDraft]);

  useRegisterDraftFlusher(commit);

  const unavailableReference = values.references.findIndex((reference) => ui.missingImages.has(reference.image_name));
  const reasons = [
    ...(unavailableReference >= 0
      ? [t('widgets.instruct.referenceUnavailable', { number: getReferenceNumber(unavailableReference) })]
      : []),
    ...getInstructValidationReasons({ ...values, instruction: text }, undefined),
  ];
  const canSend = !isLocked && !isTipUnavailable && reasons.length === 0;
  const send = () => {
    if (!canSend) {
      return;
    }
    commit();
    void actions.submit();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  };
  const isContinuing = values.continueFromNodeId !== null && tip.id !== tree.path.at(-1)?.id;
  const fallback = !isTipUnavailable || isLocked ? undefined : getNearestCompletedAncestor(tree, tip);

  return (
    <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" maxW={CHAT_MAX_WIDTH} mx="auto" p="3" w="full">
      {cancelDialog}
      {isLocked ? (
        <HStack gap="2">
          <Spinner size="md" />
          <Text color="fg.muted" flex="1" fontSize="sm">
            {ui.stepRuns[tip.stepId] === 'running' ? t('widgets.instruct.generating') : t('widgets.instruct.queued')}
          </Text>
          <Button size="xs" variant="outline" onClick={() => cancel(tip.stepId, ui.stepRuns[tip.stepId])}>
            {t('common.cancel')}
          </Button>
        </HStack>
      ) : isTipUnavailable ? (
        <HStack gap="2">
          <Text color="fg.warning" flex="1" fontSize="sm" role="status">
            {t('widgets.instruct.tipUnavailable')}
          </Text>
          {fallback ? (
            <Button
              size="xs"
              variant="outline"
              onClick={() => actions.patchValues({ continueFromNodeId: fallback.id })}
            >
              {t('widgets.instruct.continueFromPrevious')}
            </Button>
          ) : null}
        </HStack>
      ) : null}
      {isContinuing ? (
        <HStack bg="bg.subtle" gap="2" p="1" pe="2" rounded="md">
          {tip.imageName ? (
            <Image alt="" boxSize="8" objectFit="cover" rounded="sm" src={galleryImageUrls.thumbnail(tip.imageName)} />
          ) : null}
          <Text flex="1" fontSize="xs">
            {t('widgets.instruct.continuingFrom')}
          </Text>
          <IconButton
            aria-label={t('widgets.instruct.continueFromEnd')}
            size="xs"
            tooltip={t('widgets.instruct.continueFromEnd')}
            variant="ghost"
            onClick={() => actions.patchValues({ continueFromNodeId: null })}
          >
            <XIcon />
          </IconButton>
        </HStack>
      ) : null}
      <HStack align="flex-end" gap="2">
        <Textarea
          aria-disabled={isLocked || undefined}
          aria-label={t('widgets.instruct.instruction')}
          autoresize
          flex="1"
          maxH="40"
          minH="10"
          placeholder={t('widgets.instruct.placeholder')}
          // Read-only while a step runs, not disabled: the field keeps focus, so typing resumes where it was.
          readOnly={isLocked}
          resize="none"
          rows={1}
          value={text}
          onBlur={commit}
          onChange={(event) => {
            draftRef.current = event.target.value;
            setDraft(event.target.value);
            if (timer.current) {
              clearTimeout(timer.current);
            }
            timer.current = setTimeout(commit, COMMIT_DELAY_MS);
          }}
          onKeyDown={onKeyDown}
        />
        <IconButton
          aria-label={t('widgets.instruct.send')}
          disabled={!canSend}
          tooltip={t('widgets.instruct.send')}
          unavailableReason={!canSend && !isLocked ? reasons[0] : undefined}
          onClick={send}
        >
          <SendHorizontalIcon />
        </IconButton>
      </HStack>
      <Text color="fg.muted" fontSize="xs">
        {t('widgets.instruct.composerHint')}
      </Text>
    </Stack>
  );
};

const Conversation = ({ tree }: { tree: InstructTreeView }) => {
  const { t } = useTranslation();
  const { stepRuns } = useInstructUi();
  const { continueFromNodeId } = useInstructValues();
  const [jump, setJump] = useState<{ nodeId: string; token: number } | null>(null);
  const onJump = useCallback(
    (nodeId: string) => setJump((previous) => ({ nodeId, token: (previous?.token ?? 0) + 1 })),
    []
  );
  const { dialog, request } = useStartFromImage(onJump);
  const [isStartOpen, setIsStartOpen] = useState(false);
  const steps = useMemo(() => getInstructSteps(tree), [tree]);
  const root = tree.path[0]!;
  const lastKey = `${steps.length}:${steps.at(-1)?.stepId ?? ''}`;

  return (
    <>
      {/* The center view's switcher floats over the top, so the history starts below it. */}
      <Flex data-instruct-history flex="1" minH="0" overflowY="auto" pb="3" pt="14" px="4">
        <Stack gap="5" maxW={CHAT_MAX_WIDTH} mx="auto" w="full">
          <HStack justify="space-between">
            <Text fontSize="sm" fontWeight="600">
              {t('widgets.instruct.sourceImage')}
            </Text>
            <Button
              aria-expanded={isStartOpen}
              size="xs"
              variant="ghost"
              onClick={() => setIsStartOpen((open) => !open)}
            >
              {t('widgets.instruct.startOver')}
            </Button>
          </HStack>
          {isStartOpen ? (
            <GalleryMediaSlot
              accept={ACCEPT_IMAGES}
              dropId="instruct-restart-slot"
              labels={{ choose: t('widgets.instruct.chooseImage'), drop: t('widgets.instruct.dropImage') }}
              value={null}
              onChange={(item) => {
                setIsStartOpen(false);
                request(toImage(item));
              }}
            />
          ) : null}
          <Box maxW="360px">
            <VariantTile
              isContinueTarget={continueFromNodeId === root.id}
              node={root}
              parentId={root.id}
              run={undefined}
            />
          </Box>
          {steps.map((step) => (
            <StepMessage
              key={step.stepId + step.parent.id}
              continueFromNodeId={continueFromNodeId}
              run={stepRuns[step.stepId]}
              step={step}
            />
          ))}
          <Box key={lastKey} ref={scrollIntoViewOnMount} />
          {/* After the end anchor, so a jump that also changes the branch end scrolls to its node last. */}
          {jump ? <ScrollToNode key={jump.token} nodeId={jump.nodeId} /> : null}
        </Stack>
      </Flex>
      <Composer tree={tree} />
      {dialog}
    </>
  );
};

/** The Edit center view: the history of the shown branch with the instruction field under it. */
export const InstructChatView = () => {
  const { projectId, tree } = useInstructUi();

  useEnsureInstructCatalog();

  return (
    <Flex direction="column" h="full" minH="0" w="full">
      {tree ? <Conversation tree={tree} /> : <EmptyView key={projectId} />}
    </Flex>
  );
};
