Require Import ZArith.
From QuickChick Require Import QuickChick.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import QcNotation.
Import MonadNotation.
From Coq Require Import List.
Import ListNotations.

From RBT Require Import Impl Spec.

Inductive CtorColor :=
  | CtorColor_R
  | CtorColor_B.

Inductive LeafCtorColor :=
  | LeafCtorColor_R
  | LeafCtorColor_B.

Inductive CtorColorKVTree :=
  | CtorColorKVTree_E
  | CtorColorKVTree_T.

Inductive LeafCtorColorKVTree :=
  | LeafCtorColorKVTree_E.

Inductive TupLeafCtorColorCtorColorKVTreeCtorColorKVTree :=
  | MkLeafCtorColorCtorColorKVTreeCtorColorKVTree : LeafCtorColor -> CtorColorKVTree -> CtorColorKVTree -> TupLeafCtorColorCtorColorKVTreeCtorColorKVTree.

Inductive TupLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree :=
  | MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree : LeafCtorColor -> LeafCtorColorKVTree -> LeafCtorColorKVTree -> TupLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree.

Definition genLeafColor (chosen_ctor : LeafCtorColor) (stack1 : nat) (stack2 : nat) : G (Color) :=
  match chosen_ctor with
  | LeafCtorColor_R  => 
    (returnGen (R ))
  | LeafCtorColor_B  => 
    (returnGen (B ))
  end.

Definition genLeafTree (chosen_ctor : LeafCtorColorKVTree) (stack1 : nat) (stack2 : nat) : G (Tree) :=
  match chosen_ctor with
  | LeafCtorColorKVTree_E  => 
    (returnGen (E ))
  end.

Fixpoint genTree (size : nat) (chosen_ctor : CtorColorKVTree) (stack1 : nat) (stack2 : nat) : G (Tree) :=
  match size with
  | O  => match chosen_ctor with
    | CtorColorKVTree_E  => 
      (returnGen (E ))
    | CtorColorKVTree_T  => 
      (bindGen 
      (* Frequency2 *) (freq [
        (* 1 *) (match (stack1, stack2) with
        | (4, 4) => 70
        | (4, 6) => 60
        | (6, 4) => 63
        | (6, 6) => 62
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree (LeafCtorColor_R ) (LeafCtorColorKVTree_E ) (LeafCtorColorKVTree_E )))); 
        (* 2 *) (match (stack1, stack2) with
        | (4, 4) => 52
        | (4, 6) => 64
        | (6, 4) => 62
        | (6, 6) => 65
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree (LeafCtorColor_B ) (LeafCtorColorKVTree_E ) (LeafCtorColorKVTree_E ))))]) 
      (fun param_variantis => (let '(MkLeafCtorColorLeafCtorColorKVTreeLeafCtorColorKVTree ctor1 ctor2 ctor3) := param_variantis in

        (bindGen (genLeafColor ctor1 stack2 1) 
        (fun p1 => 
          (bindGen (genLeafTree ctor2 stack2 3) 
          (fun p2 => 
            (bindGen 
            (* GenZ1 *)
            (let _weight_1 := match (stack1, stack2) with
            | (4, 4) => 61
            | (4, 6) => 44
            | (6, 4) => 78
            | (6, 6) => 66
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (stack1, stack2) with
            | (4, 4) => 54
            | (4, 6) => 59
            | (6, 4) => 36
            | (6, 6) => 21
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (stack1, stack2) with
            | (4, 4) => 73
            | (4, 6) => 46
            | (6, 4) => 45
            | (6, 6) => 31
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_4, returnGen 4%Z);
              (100-_weight_4, returnGen 0%Z)
            ]) (fun n4 =>
              returnGen (n1 + n2 + n4)%Z
            )))))) 
            (fun p3 => 
              (bindGen 
              (* GenZ3 *)
              (let _weight_1 := match (stack1, stack2) with
              | (4, 4) => 28
              | (4, 6) => 54
              | (6, 4) => 37
              | (6, 6) => 30
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_1, returnGen 1%Z);
                (100-_weight_1, returnGen 0%Z)
              ]) (fun n1 =>
              (let _weight_2 := match (stack1, stack2) with
              | (4, 4) => 60
              | (4, 6) => 46
              | (6, 4) => 27
              | (6, 6) => 62
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_2, returnGen 2%Z);
                (100-_weight_2, returnGen 0%Z)
              ]) (fun n2 =>
              (let _weight_4 := match (stack1, stack2) with
              | (4, 4) => 47
              | (4, 6) => 65
              | (6, 4) => 35
              | (6, 6) => 75
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_4, returnGen 4%Z);
                (100-_weight_4, returnGen 0%Z)
              ]) (fun n4 =>
                returnGen (n1 + n2 + n4)%Z
              )))))) 
              (fun p4 => 
                (bindGen (genLeafTree ctor3 stack2 5) 
                (fun p5 => 
                  (returnGen (T p1 p2 p3 p4 p5)))))))))))))))
    end
  | S size1 => match chosen_ctor with
    | CtorColorKVTree_E  => 
      (returnGen (E ))
    | CtorColorKVTree_T  => 
      (bindGen 
      (* Frequency3 *) (freq [
        (* 1 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 1
        | (1, 4, 6) => 1
        | (1, 6, 4) => 1
        | (1, 6, 6) => 1
        | (2, 0, 4) => 0
        | (2, 0, 6) => 0
        | (3, 0, 0) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_E ) (CtorColorKVTree_E )))); 
        (* 2 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 2
        | (1, 4, 6) => 1
        | (1, 6, 4) => 1
        | (1, 6, 6) => 1
        | (2, 0, 4) => 0
        | (2, 0, 6) => 0
        | (3, 0, 0) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_E ) (CtorColorKVTree_E )))); 
        (* 3 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 4
        | (1, 4, 6) => 2
        | (1, 6, 4) => 8
        | (1, 6, 6) => 5
        | (2, 0, 4) => 0
        | (2, 0, 6) => 0
        | (3, 0, 0) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_T ) (CtorColorKVTree_E )))); 
        (* 4 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 3
        | (1, 4, 6) => 2
        | (1, 6, 4) => 4
        | (1, 6, 6) => 6
        | (2, 0, 4) => 0
        | (2, 0, 6) => 0
        | (3, 0, 0) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_T ) (CtorColorKVTree_E )))); 
        (* 5 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 4
        | (1, 4, 6) => 1
        | (1, 6, 4) => 4
        | (1, 6, 6) => 6
        | (2, 0, 4) => 0
        | (2, 0, 6) => 1
        | (3, 0, 0) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_E ) (CtorColorKVTree_T )))); 
        (* 6 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 2
        | (1, 4, 6) => 2
        | (1, 6, 4) => 4
        | (1, 6, 6) => 3
        | (2, 0, 4) => 1
        | (2, 0, 6) => 0
        | (3, 0, 0) => 0
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_E ) (CtorColorKVTree_T )))); 
        (* 7 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 93
        | (1, 4, 6) => 95
        | (1, 6, 4) => 91
        | (1, 6, 6) => 92
        | (2, 0, 4) => 95
        | (2, 0, 6) => 94
        | (3, 0, 0) => 95
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_R ) (CtorColorKVTree_T ) (CtorColorKVTree_T )))); 
        (* 8 *) (match (size, stack1, stack2) with
        | (1, 4, 4) => 92
        | (1, 4, 6) => 91
        | (1, 6, 4) => 94
        | (1, 6, 6) => 94
        | (2, 0, 4) => 95
        | (2, 0, 6) => 96
        | (3, 0, 0) => 96
        | _ => 500
        end,
        (returnGen (MkLeafCtorColorCtorColorKVTreeCtorColorKVTree (LeafCtorColor_B ) (CtorColorKVTree_T ) (CtorColorKVTree_T ))))]) 
      (fun param_variantis => (let '(MkLeafCtorColorCtorColorKVTreeCtorColorKVTree ctor1 ctor2 ctor3) := param_variantis in

        (bindGen (genLeafColor ctor1 stack2 2) 
        (fun p1 => 
          (bindGen (genTree size1 ctor2 stack2 4) 
          (fun p2 => 
            (bindGen 
            (* GenZ2 *)
            (let _weight_1 := match (size, stack1, stack2) with
            | (1, 4, 4) => 60
            | (1, 4, 6) => 69
            | (1, 6, 4) => 62
            | (1, 6, 6) => 29
            | (2, 0, 4) => 71
            | (2, 0, 6) => 64
            | (3, 0, 0) => 37
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_1, returnGen 1%Z);
              (100-_weight_1, returnGen 0%Z)
            ]) (fun n1 =>
            (let _weight_2 := match (size, stack1, stack2) with
            | (1, 4, 4) => 54
            | (1, 4, 6) => 47
            | (1, 6, 4) => 37
            | (1, 6, 6) => 44
            | (2, 0, 4) => 17
            | (2, 0, 6) => 58
            | (3, 0, 0) => 69
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_2, returnGen 2%Z);
              (100-_weight_2, returnGen 0%Z)
            ]) (fun n2 =>
            (let _weight_4 := match (size, stack1, stack2) with
            | (1, 4, 4) => 19
            | (1, 4, 6) => 50
            | (1, 6, 4) => 29
            | (1, 6, 6) => 77
            | (2, 0, 4) => 48
            | (2, 0, 6) => 27
            | (3, 0, 0) => 28
            | _ => 500
            end
            in
            bindGen (freq [
              (_weight_4, returnGen 4%Z);
              (100-_weight_4, returnGen 0%Z)
            ]) (fun n4 =>
              returnGen (n1 + n2 + n4)%Z
            )))))) 
            (fun p3 => 
              (bindGen 
              (* GenZ4 *)
              (let _weight_1 := match (size, stack1, stack2) with
              | (1, 4, 4) => 65
              | (1, 4, 6) => 42
              | (1, 6, 4) => 49
              | (1, 6, 6) => 71
              | (2, 0, 4) => 40
              | (2, 0, 6) => 87
              | (3, 0, 0) => 25
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_1, returnGen 1%Z);
                (100-_weight_1, returnGen 0%Z)
              ]) (fun n1 =>
              (let _weight_2 := match (size, stack1, stack2) with
              | (1, 4, 4) => 31
              | (1, 4, 6) => 66
              | (1, 6, 4) => 33
              | (1, 6, 6) => 54
              | (2, 0, 4) => 55
              | (2, 0, 6) => 46
              | (3, 0, 0) => 52
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_2, returnGen 2%Z);
                (100-_weight_2, returnGen 0%Z)
              ]) (fun n2 =>
              (let _weight_4 := match (size, stack1, stack2) with
              | (1, 4, 4) => 65
              | (1, 4, 6) => 75
              | (1, 6, 4) => 79
              | (1, 6, 6) => 64
              | (2, 0, 4) => 68
              | (2, 0, 6) => 70
              | (3, 0, 0) => 37
              | _ => 500
              end
              in
              bindGen (freq [
                (_weight_4, returnGen 4%Z);
                (100-_weight_4, returnGen 0%Z)
              ]) (fun n4 =>
                returnGen (n1 + n2 + n4)%Z
              )))))) 
              (fun p4 => 
                (bindGen (genTree size1 ctor3 stack2 6) 
                (fun p5 => 
                  (returnGen (T p1 p2 p3 p4 p5)))))))))))))))
    end
  end.

Definition gSized :=

  (bindGen 
  (* Frequency1 *) (freq [
    (* 1 *) (match (tt) with
    | tt => 0
    end,
    (returnGen (CtorColorKVTree_E ))); 
    (* 2 *) (match (tt) with
    | tt => 89
    end,
    (returnGen (CtorColorKVTree_T )))]) 
  (fun init_ctor => (genTree 3 init_ctor 0 0))).

(* --------------------- Tests --------------------- *)

Definition test_prop_InsertValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        (prop_InsertValid t k v)))).

(*! QuickChick test_prop_InsertValid. *)

Definition test_prop_DeleteValid :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
        prop_DeleteValid t k)).

(*! QuickChick test_prop_DeleteValid. *)

Definition test_prop_InsertPost :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
     forAll arbitrary (fun v =>
        prop_InsertPost t k k' v)))).

(*! QuickChick test_prop_InsertPost. *)

Definition test_prop_DeletePost := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeletePost t k k'))).

(*! QuickChick test_prop_DeletePost. *)
    
Definition test_prop_InsertModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun v =>
        prop_InsertModel t k v))).

(*! QuickChick test_prop_InsertModel. *)
    
Definition test_prop_DeleteModel :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
            prop_DeleteModel t k)).

(*! QuickChick test_prop_DeleteModel. *)

Definition test_prop_InsertInsert :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
    forAll arbitrary (fun v' =>     
        prop_InsertInsert t k k' v v'))))).

(*! QuickChick test_prop_InsertInsert. *)
    
Definition test_prop_InsertDelete := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v =>
        prop_InsertDelete t k k' v)))).

(*! QuickChick test_prop_InsertDelete. *)
    
Definition test_prop_DeleteInsert := 
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
    forAll arbitrary (fun v' =>
        prop_DeleteInsert t k k' v')))).

(*! QuickChick test_prop_DeleteInsert. *)
    
Definition test_prop_DeleteDelete :=  
    forAll gSized (fun t =>    
    forAll arbitrary (fun k =>
    forAll arbitrary (fun k' =>
        prop_DeleteDelete t k k'))).

(*! QuickChick test_prop_DeleteDelete. *)
          
