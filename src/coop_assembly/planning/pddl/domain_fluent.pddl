(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)

    (Assembled ?e)
    (Removed ?e)

    (PlaceAction ?r ?e ?t)
    (MoveAction ?r ?q1 ?q2 ?t2)
    (Grounded ?e)
    (Connected ?e)
    (Joined ?e1 ?e2)
    (Traj ?r ?t)

    (CanMove ?r)
    (Conf ?r ?q)
    (AtConf ?r ?q)
    (AtStart ?q ?t)
    (Assigned ?r ?e)
    (Order ?e1 ?e2)
    ; (Stiff)
  )

  ; place = remove the element
  (:action place
    :parameters (?r ?e ?t)
    :precondition (and
                       (PlaceAction ?r ?e ?t)
                       (Assembled ?e)
                       ; (Stiff)
                       (Connected ?e)
                       )
    :effect (and (Removed ?e)
                 (not (Assembled ?e))
                 )
  )

  (:action move
    :parameters (?r ?q1 ?q2 ?t2)
    :precondition (and
                        (Conf ?r ?q1)
                        ; (AtConf ?r ?q1)
                        (Conf ?r ?q2)
                        (Traj ?r ?t2)
                        (CanMove ?r)
                        (MoveAction ?r ?q1 ?q2 ?t2)
                        ;;; collision constraint handled by fluent
                       )
    :effect (and
                ;  (not (AtConf ?r ?q1))
                 (AtConf ?r ?q2)
                 (not (CanMove ?r)) ; switch to avoid transit forever
                 )
  )

  (:derived (Connected ?e2)
   (or (Grounded ?e2)
       (exists (?e1) (and (Joined ?e1 ?e2)
                          (Assembled ?e1)
                          (Connected ?e1)
                     )
       )
   )
  )
)
